# copy from nemo
"""LLM Rails entry point."""

import asyncio
import importlib.util
import json
import logging
import os
import re
import threading
import time
from functools import partial
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Type, Union, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

from nemoguardrails.actions.llm.generation import LLMGenerationActions
from nemoguardrails.actions.llm.utils import (
    get_and_clear_reasoning_trace_contextvar,
    get_colang_history,
)
from nemoguardrails.actions.output_mapping import is_output_blocked
from nemoguardrails.actions.v2_x.generation import LLMGenerationActionsV2dotx
from nemoguardrails.colang import parse_colang_file
from nemoguardrails.colang.v1_0.runtime.flows import compute_context
from nemoguardrails.colang.v1_0.runtime.runtime import Runtime, RuntimeV1_0
from nemoguardrails.colang.v2_x.runtime.flows import Action, State
from nemoguardrails.colang.v2_x.runtime.runtime import RuntimeV2_x
from nemoguardrails.colang.v2_x.runtime.serialization import (
    json_to_state,
    state_to_json,
)
from nemoguardrails.context import (
    explain_info_var,
    generation_options_var,
    llm_stats_var,
    raw_llm_request,
    streaming_handler_var,
)
from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.embeddings.providers import register_embedding_provider
from nemoguardrails.embeddings.providers.base import EmbeddingModel
from nemoguardrails.kb.kb import KnowledgeBase
from nemoguardrails.llm.models.initializer import (
    ModelInitializationError,
    init_llm_model,
)
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.logging.processing_log import compute_generation_log
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.logging.verbose import set_verbose
from nemoguardrails.patch_asyncio import check_sync_call_from_async_loop
from nemoguardrails.rails.llm.buffer import get_buffer_strategy
from nemoguardrails.rails.llm.config import EmbeddingSearchProvider, Model, RailsConfig
from nemoguardrails.rails.llm.options import (
    GenerationLog,
    GenerationOptions,
    GenerationResponse,
)
from nemoguardrails.rails.llm.utils import get_history_cache_key
from nemoguardrails.streaming import END_OF_STREAM, StreamingHandler
from nemoguardrails.utils import (
    extract_error_json,
    get_or_create_event_loop,
    new_event_dict,
    new_uuid,
)

log = logging.getLogger(__name__)

process_events_semaphore = asyncio.Semaphore(1)


class LLMRails:
    """Rails based on a given configuration."""

    config: RailsConfig
    llm: Optional[Union[BaseLLM, BaseChatModel]]
    runtime: Runtime

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        verbose: bool = False,
    ):
        """Initializes the LLMRails instance.

        Args:
            config: A rails configuration.
            llm: An optional LLM engine to use. If provided, this will be used as the main LLM
                and will take precedence over any main LLM specified in the config.
            verbose: Whether the logging should be verbose or not.
        """
        self.config = config
        self.llm = llm
        self.verbose = verbose

        if self.verbose:
            set_verbose(True, llm_calls=True)

        # We allow the user to register additional embedding search providers, so we keep
        # an index of them.
        self.embedding_search_providers = {}

        # The default embeddings model is using FastEmbed
        self.default_embedding_model = "all-MiniLM-L6-v2"
        self.default_embedding_engine = "FastEmbed"
        self.default_embedding_params = {}

        # We keep a cache of the events history associated with a sequence of user messages.
        # TODO: when we update the interface to allow to return a "state object", this
        #   should be removed
        self.events_history_cache = {}

        # Weather the main LLM supports streaming
        self.main_llm_supports_streaming = False

        # We also load the default flows from the `default_flows.yml` file in the current folder.
        # But only for version 1.0.
        # TODO: decide on the default flows for 2.x.
        if config.colang_version == "1.0":
            # We also load the default flows from the `llm_flows.co` file in the current folder.
            current_folder = os.path.dirname(__file__)
            default_flows_file = "llm_flows.co"
            default_flows_path = os.path.join(current_folder, default_flows_file)
            with open(default_flows_path, "r") as f:
                default_flows_content = f.read()
                default_flows = parse_colang_file(
                    default_flows_file, default_flows_content
                )["flows"]

            # We mark all the default flows as system flows.
            for flow_config in default_flows:
                flow_config["is_system_flow"] = True

            # We add the default flows to the config.
            self.config.flows.extend(default_flows)

            # We also need to load the content from the components library.
            library_path = os.path.join(os.path.dirname(__file__), "../../library")
            for root, dirs, files in os.walk(library_path):
                for file in files:
                    # Extract the full path for the file
                    full_path = os.path.join(root, file)
                    if file.endswith(".co"):
                        log.debug(f"Loading file: {full_path}")
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = parse_colang_file(
                                file, content=f.read(), version=config.colang_version
                            )
                            if not content:
                                continue

                        # We mark all the flows coming from the guardrails library as system flows.
                        for flow_config in content["flows"]:
                            flow_config["is_system_flow"] = True

                        # We load all the flows
                        self.config.flows.extend(content["flows"])

                        # And all the messages as well, if they have not been overwritten
                        for message_id, utterances in content.get(
                            "bot_messages", {}
                        ).items():
                            if message_id not in self.config.bot_messages:
                                self.config.bot_messages[message_id] = utterances

        # Last but not least, we mark all the flows that are used in any of the rails
        # as system flows (so they don't end up in the prompt).

        rail_flow_ids = (
            config.rails.input.flows
            + config.rails.output.flows
            + config.rails.retrieval.flows
        )

        for flow_config in self.config.flows:
            if flow_config.get("id") in rail_flow_ids:
                flow_config["is_system_flow"] = True

                # We also mark them as subflows by default, to simplify the syntax
                flow_config["is_subflow"] = True

        # We check if the configuration or any of the imported ones have config.py modules.
        config_modules = []
        for _path in list(self.config.imported_paths.values()) + [
            self.config.config_path
        ]:
            if _path:
                filepath = os.path.join(_path, "config.py")
                if os.path.exists(filepath):
                    filename = os.path.basename(filepath)
                    spec = importlib.util.spec_from_file_location(filename, filepath)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    config_modules.append(config_module)

        # First, we initialize the runtime.
        if config.colang_version == "1.0":
            self.runtime = RuntimeV1_0(config=config, verbose=verbose)
        elif config.colang_version == "2.x":
            self.runtime = RuntimeV2_x(config=config, verbose=verbose)
        else:
            raise ValueError(f"Unsupported colang version: {config.colang_version}.")

        # If we have a config_modules with an `init` function, we call it.
        # We need to call this here because the `init` might register additional
        # LLM providers.
        for config_module in config_modules:
            if hasattr(config_module, "init"):
                config_module.init(self)

        # If we have a customized embedding model, we'll use it.
        for model in self.config.models:
            if model.type == "embeddings":
                self.default_embedding_model = model.model
                self.default_embedding_engine = model.engine
                self.default_embedding_params = model.parameters or {}
                break

        # InteractionLogAdapters used for tracing
        # We ensure that it is used after config.py is loaded
        if config.tracing:
            from nemoguardrails.tracing import create_log_adapters

            self._log_adapters = create_log_adapters(config.tracing)

        # We run some additional checks on the config
        self._validate_config()

        # Next, we initialize the LLM engines (main engine and action engines if specified).
        self._init_llms()

        # Next, we initialize the LLM Generate actions and register them.
        llm_generation_actions_class = (
            LLMGenerationActions
            if config.colang_version == "1.0"
            else LLMGenerationActionsV2dotx
        )
        self.llm_generation_actions = llm_generation_actions_class(
            config=config,
            llm=self.llm,
            llm_task_manager=self.runtime.llm_task_manager,
            get_embedding_search_provider_instance=self._get_embeddings_search_provider_instance,
            verbose=verbose,
        )

        # If there's already an action registered, we don't override.
        self.runtime.register_actions(self.llm_generation_actions, override=False)

        # Next, we initialize the Knowledge Base
        # There are still some edge cases not covered by nest_asyncio.
        # Using a separate thread always for now.
        loop = get_or_create_event_loop()
        if True or check_sync_call_from_async_loop():
            t = threading.Thread(target=asyncio.run, args=(self._init_kb(),))
            t.start()
            t.join()
        else:
            loop.run_until_complete(self._init_kb())

        # We also register the kb as a parameter that can be passed to actions.
        self.runtime.register_action_param("kb", self.kb)

        # Reference to the general ExplainInfo object.
        self.explain_info = None

    def update_llm(self, llm):
        """Replace the main LLM with the provided one.

        Arguments:
            llm: The new LLM that should be used.
        """
        self.llm = llm
        self.llm_generation_actions.llm = llm
        self.runtime.register_action_param("llm", llm)

    def _validate_config(self):
        """Runs additional validation checks on the config."""

        if self.config.colang_version == "1.0":
            existing_flows_names = set([flow.get("id") for flow in self.config.flows])
        else:
            existing_flows_names = set([flow.get("name") for flow in self.config.flows])

        for flow_name in self.config.rails.input.flows:
            # content safety check input/output flows are special as they have parameters
            if flow_name.startswith("content safety check") or flow_name.startswith(
                "topic safety check"
            ):
                continue
            if flow_name not in existing_flows_names:
                raise ValueError(
                    f"The provided input rail flow `{flow_name}` does not exist"
                )

        for flow_name in self.config.rails.output.flows:
            if flow_name.startswith("content safety check") or flow_name.startswith(
                "topic safety check"
            ):
                continue
            if flow_name not in existing_flows_names:
                raise ValueError(
                    f"The provided output rail flow `{flow_name}` does not exist"
                )

        for flow_name in self.config.rails.retrieval.flows:
            if flow_name not in existing_flows_names:
                raise ValueError(
                    f"The provided retrieval rail flow `{flow_name}` does not exist"
                )

        # If both passthrough mode and single call mode are specified, we raise an exception.
        if self.config.passthrough and self.config.rails.dialog.single_call.enabled:
            raise ValueError(
                "The passthrough mode and the single call dialog rails mode can't be used at the same time. "
                "The single call mode needs to use an altered prompt when prompting the LLM. "
            )

    async def _init_kb(self):
        """Initializes the knowledge base."""
        self.kb = None

        if not self.config.docs:
            return

        documents = [doc.content for doc in self.config.docs]
        self.kb = KnowledgeBase(
            documents=documents,
            config=self.config.knowledge_base,
            get_embedding_search_provider_instance=self._get_embeddings_search_provider_instance,
        )
        self.kb.init()
        await self.kb.build()

    def _prepare_model_kwargs(self, model_config):
        """
        Prepare kwargs for model initialization, including API key from environment variable.

        Args:
            model_config: The model configuration object

        Returns:
            dict: The prepared kwargs for model initialization
        """
        kwargs = model_config.parameters or {}

        # If the optional API Key Environment Variable is set, add it to kwargs
        if model_config.api_key_env_var:
            api_key = os.environ.get(model_config.api_key_env_var)
            if api_key:
                kwargs["api_key"] = api_key

        return kwargs

    def _configure_main_llm_streaming(
        self,
        llm: Union[BaseLLM, BaseChatModel],
        model_name: Optional[str] = None,
        provider_name: Optional[str] = None,
    ):
        """Configure streaming support for the main LLM.

        Args:
            llm (Union[BaseLLM, BaseChatModel]): The main LLM model instance.
            model_name (Optional[str], optional): Optional model name for logging.
            provider_name (Optional[str], optional): Optional provider name for logging.

        """
        if not self.config.streaming:
            return

        if "streaming" in llm.model_fields:
            llm.streaming = True
            self.main_llm_supports_streaming = True
        else:
            self.main_llm_supports_streaming = False
            if model_name and provider_name:
                log.warning(
                    "Model %s from provider %s does not support streaming.",
                    model_name,
                    provider_name,
                )
            else:
                log.warning("Provided main LLM does not support streaming.")

    def _init_llms(self):
        """
        Initializes the right LLM engines based on the configuration.
        There can be multiple LLM engines and types that can be specified in the config.
        The main LLM engine is the one that will be used for all the core guardrails generations.
        Other LLM engines can be specified for use in specific actions.

        The reason we provide an option for decoupling the main LLM engine from the action LLM
        is to allow for flexibility in using specialized LLM engines for specific actions.

        Raises:
            ModelInitializationError: If any model initialization fails
        """
        # If the user supplied an already-constructed LLM via the constructor we
        # treat it as the *main* model, but **still** iterate through the
        # configuration to load any additional models (e.g. `content_safety`).

        if self.llm:
            # If an LLM was provided via constructor, use it as the main LLM
            # Log a warning if a main LLM is also specified in the config
            if any(model.type == "main" for model in self.config.models):
                log.warning(
                    "Both an LLM was provided via constructor and a main LLM is specified in the config. "
                    "The LLM provided via constructor will be used and the main LLM from config will be ignored."
                )
            self.runtime.register_action_param("llm", self.llm)

            self._configure_main_llm_streaming(self.llm)
        else:
            # Otherwise, initialize the main LLM from the config
            main_model = next(
                (model for model in self.config.models if model.type == "main"), None
            )

            if main_model:
                kwargs = self._prepare_model_kwargs(main_model)
                self.llm = init_llm_model(
                    model_name=main_model.model,
                    provider_name=main_model.engine,
                    mode="chat",
                    kwargs=kwargs,
                )
                self.runtime.register_action_param("llm", self.llm)

                self._configure_main_llm_streaming(
                    self.llm,
                    model_name=main_model.model,
                    provider_name=main_model.engine,
                )
            else:
                log.warning(
                    "No main LLM specified in the config and no LLM provided via constructor."
                )

        llms = dict()

        for llm_config in self.config.models:
            if llm_config.type == "embeddings":
                continue

            # If a constructor LLM is provided, skip initializing any 'main' model from config
            if self.llm and llm_config.type == "main":
                continue

            try:
                model_name = llm_config.model
                provider_name = llm_config.engine
                kwargs = self._prepare_model_kwargs(llm_config)
                mode = llm_config.mode

                llm_model = init_llm_model(
                    model_name=model_name,
                    provider_name=provider_name,
                    mode=mode,
                    kwargs=kwargs,
                )

                if llm_config.type == "main":
                    # If a main LLM was already injected, skip creating another
                    # one. Otherwise, create and register it.
                    if not self.llm:
                        self.llm = llm_model
                        self.runtime.register_action_param("llm", self.llm)
                else:
                    model_name = f"{llm_config.type}_llm"
                    if not hasattr(self, model_name):
                        setattr(self, model_name, llm_model)
                    self.runtime.register_action_param(
                        model_name, getattr(self, model_name)
                    )
                    # this is used for content safety and topic control
                    llms[llm_config.type] = getattr(self, model_name)

            except ModelInitializationError as e:
                log.error("Failed to initialize model: %s", str(e))
                raise
            except Exception as e:
                log.error("Unexpected error initializing model: %s", str(e))
                raise

        self.runtime.register_action_param("llms", llms)

    def _get_embeddings_search_provider_instance(
        self, esp_config: Optional[EmbeddingSearchProvider] = None
    ) -> EmbeddingsIndex:
        if esp_config is None:
            esp_config = EmbeddingSearchProvider()

        if esp_config.name == "default":
            from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex

            return BasicEmbeddingsIndex(
                embedding_model=esp_config.parameters.get(
                    "embedding_model", self.default_embedding_model
                ),
                embedding_engine=esp_config.parameters.get(
                    "embedding_engine", self.default_embedding_engine
                ),
                embedding_params=esp_config.parameters.get(
                    "embedding_parameters", self.default_embedding_params
                ),
                cache_config=esp_config.cache,
                # We make sure we also pass additional relevant params.
                **{
                    k: v
                    for k, v in esp_config.parameters.items()
                    if k
                    in [
                        "use_batching",
                        "max_batch_size",
                        "matx_batch_hold",
                        "search_threshold",
                    ]
                    and v is not None
                },
            )
        else:
            if esp_config.name not in self.embedding_search_providers:
                raise Exception(f"Unknown embedding search provider: {esp_config.name}")
            else:
                kwargs = esp_config.parameters
                return self.embedding_search_providers[esp_config.name](**kwargs)

    def _get_events_for_messages(self, messages: List[dict], state: Any):
        """Return the list of events corresponding to the provided messages.

        Tries to find a prefix of messages for which we have already a list of events
        in the cache. For the rest, they are converted as is.

        The reason this cache exists is that we want to benefit from events generated in
        previous turns, which can't be computed again because it would be expensive (e.g.,
        involving multiple LLM calls).

        When an explicit state object will be added, this mechanism can be removed.

        Args:
            messages: The list of messages.

        Returns:
            A list of events.
        """
        events = []

        if self.config.colang_version == "1.0":
            # We try to find the longest prefix of messages for which we have a cache
            # of events.
            p = len(messages) - 1
            while p > 0:
                cache_key = get_history_cache_key(messages[0:p])
                if cache_key in self.events_history_cache:
                    events = self.events_history_cache[cache_key].copy()
                    break

                p -= 1

            # For the rest of the messages, we transform them directly into events.
            # TODO: Move this to separate function once more types of messages are supported.
            for idx in range(p, len(messages)):
                msg = messages[idx]
                if msg["role"] == "user":
                    events.append(
                        {
                            "type": "UtteranceUserActionFinished",
                            "final_transcript": msg["content"],
                        }
                    )

                    # If it's not the last message, we also need to add the `UserMessage` event
                    if idx != len(messages) - 1:
                        events.append(
                            {
                                "type": "UserMessage",
                                "text": msg["content"],
                            }
                        )

                elif msg["role"] == "assistant":
                    action_uid = new_uuid()
                    start_event = new_event_dict(
                        "StartUtteranceBotAction",
                        script=msg["content"],
                        action_uid=action_uid,
                    )
                    finished_event = new_event_dict(
                        "UtteranceBotActionFinished",
                        final_script=msg["content"],
                        is_success=True,
                        action_uid=action_uid,
                    )
                    events.extend([start_event, finished_event])
                elif msg["role"] == "context":
                    events.append({"type": "ContextUpdate", "data": msg["content"]})
                elif msg["role"] == "event":
                    events.append(msg["event"])
                elif msg["role"] == "system":
                    # Handle system messages - convert them to SystemMessage events
                    events.append({"type": "SystemMessage", "content": msg["content"]})
        else:
            for idx in range(len(messages)):
                msg = messages[idx]
                if msg["role"] == "user":
                    events.append(
                        {
                            "type": "UtteranceUserActionFinished",
                            "final_transcript": msg["content"],
                        }
                    )

                elif msg["role"] == "assistant":
                    raise ValueError(
                        "Providing `assistant` messages as input is not supported for Colang 2.0 configurations."
                    )
                elif msg["role"] == "context":
                    events.append({"type": "ContextUpdate", "data": msg["content"]})
                elif msg["role"] == "event":
                    events.append(msg["event"])
                elif msg["role"] == "system":
                    # Handle system messages - convert them to SystemMessage events
                    events.append({"type": "SystemMessage", "content": msg["content"]})
                elif msg["role"] == "tool":
                    action_uid = msg["tool_call_id"]
                    return_value = msg["content"]
                    action: Action = state.actions[action_uid]
                    events.append(
                        new_event_dict(
                            f"{action.name}Finished",
                            action_uid=action_uid,
                            action_name=action.name,
                            status="success",
                            is_success=True,
                            return_value=return_value,
                            events=[],
                        )
                    )

        return events

    @staticmethod
    def _ensure_explain_info() -> ExplainInfo:
        """Ensure that the ExplainInfo variable is present in the current context

        Returns:
            A ExplainInfo class containing the llm calls' statistics
        """
        explain_info = explain_info_var.get()
        if explain_info is None:
            explain_info = ExplainInfo()
            explain_info_var.set(explain_info)

        return explain_info

    async def generate_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Union[dict, State]] = None,
        streaming_handler: Optional[StreamingHandler] = None,
    ) -> Union[str, dict, GenerationResponse, Tuple[dict, dict]]:
        """Generate a completion or a next message.

        The format for messages is the following:

        ```python
            [
                {"role": "context", "content": {"user_name": "John"}},
                {"role": "user", "content": "Hello! How are you?"},
                {"role": "assistant", "content": "I am fine, thank you!"},
                {"role": "event", "event": {"type": "UserSilent"}},
                ...
            ]
        ```

        Args:
            prompt: The prompt to be used for completion.
            messages: The history of messages to be used to generate the next message.
            options: Options specific for the generation.
            state: The state object that should be used as the starting point.
            streaming_handler: If specified, and the config supports streaming, the
              provided handler will be used for streaming.

        Returns:
            The completion (when a prompt is provided) or the next message.

        System messages are not yet supported."""
        # If a state object is specified, then we switch to "generation options" mode.
        # This is because we want the output to be a GenerationResponse which will contain
        # the output state.
        if state is not None:
            # We deserialize the state if needed.
            if isinstance(state, dict) and state.get("version", "1.0") == "2.x":
                state = json_to_state(state["state"])

            if options is None:
                options = GenerationOptions()

        # We allow options to be specified both as a dict and as an object.
        if options and isinstance(options, dict):
            options = GenerationOptions(**options)

        # Save the generation options in the current async context.
        generation_options_var.set(options)

        if streaming_handler:
            streaming_handler_var.set(streaming_handler)

        # Initialize the object with additional explanation information.
        # We allow this to also be set externally. This is useful when multiple parallel
        # requests are made.
        self.explain_info = self._ensure_explain_info()

        if prompt is not None:
            # Currently, we transform the prompt request into a single turn conversation
            messages = [{"role": "user", "content": prompt}]
            raw_llm_request.set(prompt)
        else:
            raw_llm_request.set(messages)

        # If we have generation options, we also add them to the context
        if options:
            messages = [
                {"role": "context", "content": {"generation_options": options.dict()}}
            ] + messages

        # If the last message is from the assistant, rather than the user, then
        # we move that to the `$bot_message` variable. This is to enable a more
        # convenient interface. (only when dialog rails are disabled)
        if (
            messages[-1]["role"] == "assistant"
            and options
            and options.rails.dialog is False
        ):
            # We already have the first message with a context update, so we use that
            messages[0]["content"]["bot_message"] = messages[-1]["content"]
            messages = messages[0:-1]

        # TODO: Add support to load back history of events, next to history of messages
        #   This is important as without it, the LLM prediction is not as good.

        t0 = time.time()

        # Initialize the LLM stats
        llm_stats = LLMStats()
        llm_stats_var.set(llm_stats)
        processing_log = []

        # The array of events corresponding to the provided sequence of messages.
        events = self._get_events_for_messages(messages, state)

        if self.config.colang_version == "1.0":
            # If we had a state object, we also need to prepend the events from the state.
            state_events = []
            if state:
                assert isinstance(state, dict)
                state_events = state["events"]

            new_events = []
            # Compute the new events.
            try:
                new_events = await self.runtime.generate_events(
                    state_events + events, processing_log=processing_log
                )
                output_state = None

            except Exception as e:
                log.error("Error in generate_async: %s", e, exc_info=True)
                streaming_handler = streaming_handler_var.get()
                if streaming_handler:
                    # Push an error chunk instead of None.
                    error_message = str(e)
                    error_dict = extract_error_json(error_message)
                    error_payload = json.dumps(error_dict)
                    await streaming_handler.push_chunk(error_payload)
                    # push a termination signal
                    await streaming_handler.push_chunk(END_OF_STREAM)
                # Re-raise the exact exception
                raise
        else:
            # In generation mode, by default the bot response is an instant action.
            instant_actions = ["UtteranceBotAction"]
            if self.config.rails.actions.instant_actions is not None:
                instant_actions = self.config.rails.actions.instant_actions

            # Cast this explicitly to avoid certain warnings
            runtime: RuntimeV2_x = cast(RuntimeV2_x, self.runtime)

            # Compute the new events.
            # In generation mode, the processing is always blocking, i.e., it waits for
            # all local actions (sync and async).
            new_events, output_state = await runtime.process_events(
                events, state=state, instant_actions=instant_actions, blocking=True
            )
            # We also encode the output state as a JSON
            output_state = {"state": state_to_json(output_state), "version": "2.x"}

        # Extract and join all the messages from StartUtteranceBotAction events as the response.
        responses = []
        response_tool_calls = []
        response_events = []
        new_extra_events = []
        exception = None

        # The processing is different for Colang 1.0 and 2.0
        if self.config.colang_version == "1.0":
            for event in new_events:
                if event["type"] == "StartUtteranceBotAction":
                    # Check if we need to remove a message
                    if event["script"] == "(remove last message)":
                        responses = responses[0:-1]
                    else:
                        responses.append(event["script"])
                elif event["type"].endswith("Exception"):
                    exception = event

        else:
            for event in new_events:
                start_action_match = re.match(r"Start(.*Action)", event["type"])

                if start_action_match:
                    action_name = start_action_match[1]
                    # TODO: is there an elegant way to extract just the arguments?
                    arguments = {
                        k: v
                        for k, v in event.items()
                        if k != "type"
                        and k != "uid"
                        and k != "event_created_at"
                        and k != "source_uid"
                        and k != "action_uid"
                    }
                    response_tool_calls.append(
                        {
                            "id": event["action_uid"],
                            "type": "function",
                            "function": {"name": action_name, "arguments": arguments},
                        }
                    )

                elif event["type"] == "UtteranceBotActionFinished":
                    responses.append(event["final_script"])
                else:
                    # We just append the event
                    response_events.append(event)

        if exception:
            new_message = {"role": "exception", "content": exception}

        else:
            # Ensure all items in responses are strings
            responses = [
                str(response) if not isinstance(response, str) else response
                for response in responses
            ]
            new_message = {"role": "assistant", "content": "\n".join(responses)}
        if response_tool_calls:
            new_message["tool_calls"] = response_tool_calls
        if response_events:
            new_message["events"] = response_events

        if self.config.colang_version == "1.0":
            events.extend(new_events)
            events.extend(new_extra_events)

            # If a state object is not used, then we use the implicit caching
            if state is None:
                # Save the new events in the history and update the cache
                cache_key = get_history_cache_key(messages + [new_message])
                self.events_history_cache[cache_key] = events
            else:
                output_state = {"events": events}

        # If logging is enabled, we log the conversation
        # TODO: add support for logging flag
        self.explain_info.colang_history = get_colang_history(events)
        if self.verbose:
            log.info(
                f"Conversation history so far: \n{self.explain_info.colang_history}"
            )

        total_time = time.time() - t0
        log.info(
            "--- :: Total processing took %.2f seconds. LLM Stats: %s"
            % (total_time, llm_stats)
        )

        # If there is a streaming handler, we make sure we close it now
        streaming_handler = streaming_handler_var.get()
        if streaming_handler:
            # print("Closing the stream handler explicitly")
            await streaming_handler.push_chunk(END_OF_STREAM)

        # IF tracing is enabled we need to set GenerationLog attrs
        if self.config.tracing.enabled:
            if options is None:
                options = GenerationOptions()
            if (
                not options.log.activated_rails
                or not options.log.llm_calls
                or not options.log.internal_events
            ):
                options.log.activated_rails = True
                options.log.llm_calls = True
                options.log.internal_events = True

        # If we have generation options, we prepare a GenerationResponse instance.
        if options:
            # If a prompt was used, we only need to return the content of the message.
            if prompt:
                res = GenerationResponse(response=new_message["content"])
            else:
                res = GenerationResponse(response=[new_message])

            if reasoning_trace := get_and_clear_reasoning_trace_contextvar():
                if prompt:
                    res.response = reasoning_trace + res.response
                else:
                    res.response[0]["content"] = (
                        reasoning_trace + res.response[0]["content"]
                    )

            if self.config.colang_version == "1.0":
                # If output variables are specified, we extract their values
                if options.output_vars:
                    context = compute_context(events)
                    if isinstance(options.output_vars, list):
                        # If we have only a selection of keys, we filter to only that.
                        res.output_data = {
                            k: context.get(k) for k in options.output_vars
                        }
                    else:
                        # Otherwise, we return the full context
                        res.output_data = context

                _log = compute_generation_log(processing_log)

                # Include information about activated rails and LLM calls if requested
                if options.log.activated_rails or options.log.llm_calls:
                    res.log = GenerationLog()

                    # We always include the stats
                    res.log.stats = _log.stats

                    if options.log.activated_rails:
                        res.log.activated_rails = _log.activated_rails

                    if options.log.llm_calls:
                        res.log.llm_calls = []
                        for activated_rail in _log.activated_rails:
                            for executed_action in activated_rail.executed_actions:
                                res.log.llm_calls.extend(executed_action.llm_calls)

                # Include internal events if requested
                if options.log.internal_events:
                    if res.log is None:
                        res.log = GenerationLog()

                    res.log.internal_events = new_events

                # Include the Colang history if requested
                if options.log.colang_history:
                    if res.log is None:
                        res.log = GenerationLog()

                    res.log.colang_history = get_colang_history(events)

                # Include the raw llm output if requested
                if options.llm_output:
                    # Currently, we include the output from the generation LLM calls.
                    for activated_rail in _log.activated_rails:
                        if activated_rail.type == "generation":
                            for executed_action in activated_rail.executed_actions:
                                for llm_call in executed_action.llm_calls:
                                    res.llm_output = llm_call.raw_response
            else:
                if options.output_vars:
                    raise ValueError(
                        "The `output_vars` option is not supported for Colang 2.0 configurations."
                    )

                if (
                    options.log.activated_rails
                    or options.log.llm_calls
                    or options.log.internal_events
                    or options.log.colang_history
                ):
                    raise ValueError(
                        "The `log` option is not supported for Colang 2.0 configurations."
                    )

                if options.llm_output:
                    raise ValueError(
                        "The `llm_output` option is not supported for Colang 2.0 configurations."
                    )

            # Include the state
            if state is not None:
                res.state = output_state

            if self.config.tracing.enabled:
                # TODO: move it to the top once resolved circular dependency of eval
                # lazy import to avoid circular dependency
                from nemoguardrails.tracing import Tracer

                # Create a Tracer instance with instantiated adapters
                tracer = Tracer(
                    input=messages, response=res, adapters=self._log_adapters
                )
                await tracer.export_async()

            return res
        else:
            # If a prompt is used, we only return the content of the message.

            if reasoning_trace := get_and_clear_reasoning_trace_contextvar():
                new_message["content"] = reasoning_trace + new_message["content"]

            if prompt:
                return new_message["content"]
            else:
                return new_message

    def stream_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Union[dict, State]] = None,
        include_generation_metadata: Optional[bool] = False,
    ) -> AsyncIterator[str]:
        """Simplified interface for getting directly the streamed tokens from the LLM."""
        self.explain_info = self._ensure_explain_info()

        streaming_handler = StreamingHandler(
            include_generation_metadata=include_generation_metadata
        )

        # todo use a context var for buffer strategy and return it here?
        # then iterating over buffer strategy is nested loop?
        asyncio.create_task(
            self.generate_async(
                prompt=prompt,
                messages=messages,
                streaming_handler=streaming_handler,
                options=options,
                state=state,
            )
        )
        # when we have output rails we wrap the streaming handler
        # if len(self.config.rails.output.flows) > 0:
        #
        if self.config.rails.output.streaming.enabled:
            # returns an async generator
            return self._run_output_rails_in_streaming(
                streaming_handler=streaming_handler,
                messages=messages,
                prompt=prompt,
            )
        else:
            return streaming_handler

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[dict] = None,
    ):
        """Synchronous version of generate_async."""

        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "You are using the sync `generate` inside async code. "
                "You should replace with `await generate_async(...)` or use `nest_asyncio.apply()`."
            )

        loop = get_or_create_event_loop()

        return loop.run_until_complete(
            self.generate_async(
                prompt=prompt,
                messages=messages,
                options=options,
                state=state,
            )
        )

    async def generate_events_async(
        self,
        events: List[dict],
    ) -> List[dict]:
        """Generate the next events based on the provided history.

        The format for events is the following:

        ```python
            [
                {"type": "...", ...},
                ...
            ]
        ```

        Args:
            events: The history of events to be used to generate the next events.
            options: The options to be used for the generation.

        Returns:
            The newly generate event(s).

        """
        t0 = time.time()

        # Initialize the LLM stats
        llm_stats = LLMStats()
        llm_stats_var.set(llm_stats)

        # Compute the new events.
        processing_log = []
        new_events = await self.runtime.generate_events(
            events, processing_log=processing_log
        )

        # If logging is enabled, we log the conversation
        # TODO: add support for logging flag
        if self.verbose:
            history = get_colang_history(events)
            log.info(f"Conversation history so far: \n{history}")

        log.info("--- :: Total processing took %.2f seconds." % (time.time() - t0))
        log.info("--- :: Stats: %s" % llm_stats)

        return new_events

    def generate_events(
        self,
        events: List[dict],
    ) -> List[dict]:
        """Synchronous version of `LLMRails.generate_events_async`."""

        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "You are using the sync `generate_events` inside async code. "
                "You should replace with `await generate_events_async(...)` or use `nest_asyncio.apply()`."
            )

        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.generate_events_async(events=events))

    async def process_events_async(
        self,
        events: List[dict],
        state: Optional[dict] = None,
        blocking: bool = False,
    ) -> Tuple[List[dict], dict]:
        """Process a sequence of events in a given state.

        The events will be processed one by one, in the input order.

        Args:
            events: A sequence of events that needs to be processed.
            state: The state that should be used as the starting point. If not provided,
              a clean state will be used.

        Returns:
            (output_events, output_state) Returns a sequence of output events and an output
              state.
        """
        t0 = time.time()
        llm_stats = LLMStats()
        llm_stats_var.set(llm_stats)

        # Compute the new events.
        # We need to protect 'process_events' to be called only once at a time
        # TODO (cschueller): Why is this?
        async with process_events_semaphore:
            output_events, output_state = await self.runtime.process_events(
                events, state, blocking
            )

        took = time.time() - t0
        # Small tweak, disable this when there were no events (or it was just too fast).
        if took > 0.1:
            log.info("--- :: Total processing took %.2f seconds." % took)
            log.info("--- :: Stats: %s" % llm_stats)

        return output_events, output_state

    def process_events(
        self,
        events: List[dict],
        state: Optional[dict] = None,
        blocking: bool = False,
    ) -> Tuple[List[dict], dict]:
        """Synchronous version of `LLMRails.process_events_async`."""

        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "You are using the sync `generate_events` inside async code. "
                "You should replace with `await generate_events_async(...)."
            )

        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            self.process_events_async(events, state, blocking)
        )

    def register_action(self, action: callable, name: Optional[str] = None):
        """Register a custom action for the rails configuration."""
        self.runtime.register_action(action, name)

    def register_action_param(self, name: str, value: Any):
        """Registers a custom action parameter."""
        self.runtime.register_action_param(name, value)

    def register_filter(self, filter_fn: callable, name: Optional[str] = None):
        """Register a custom filter for the rails configuration."""
        self.runtime.llm_task_manager.register_filter(filter_fn, name)

    def register_output_parser(self, output_parser: callable, name: str):
        """Register a custom output parser for the rails configuration."""
        self.runtime.llm_task_manager.register_output_parser(output_parser, name)

    def register_prompt_context(self, name: str, value_or_fn: Any):
        """Register a value to be included in the prompt context.

        :name: The name of the variable or function that will be used.
        :value_or_fn: The value or function that will be used to generate the value.
        """
        self.runtime.llm_task_manager.register_prompt_context(name, value_or_fn)

    def register_embedding_search_provider(
        self, name: str, cls: Type[EmbeddingsIndex]
    ) -> None:
        """Register a new embedding search provider.

        Args:
            name: The name of the embedding search provider that will be used.
            cls: The class that will be used to generate and search embedding
        """

        self.embedding_search_providers[name] = cls

    def register_embedding_provider(
        self, cls: Type[EmbeddingModel], name: Optional[str] = None
    ) -> None:
        """Register a custom embedding provider.

        Args:
            model (Type[EmbeddingModel]): The embedding model class.
            name (str): The name of the embedding engine. If available in the model, it will be used.

        Raises:
            ValueError: If the engine name is not provided and the model does not have an engine name.
            ValueError: If the model does not have 'encode' or 'encode_async' methods.
        """
        register_embedding_provider(engine_name=name, model=cls)

    def explain(self) -> ExplainInfo:
        """Helper function to return the latest ExplainInfo object."""
        return self.explain_info

    def __getstate__(self):
        return {"config": self.config}

    def __setstate__(self, state):
        if state["config"].config_path:
            config = RailsConfig.from_path(state["config"].config_path)
        else:
            config = state["config"]
        self.__init__(config=config, verbose=False)

    async def _run_output_rails_in_streaming(
        self,
        streaming_handler: AsyncIterator[str],
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        stream_first: Optional[bool] = None,
    ) -> AsyncIterator[str]:
        """
        1. Buffers tokens from 'streaming_handler' via BufferStrategy.
        2. Runs sequential (parallel for colang 2.0 in future) flows for each chunk.
        3. Yields the chunk if not blocked, or STOP if blocked.
        """

        def _get_last_context_message(
            messages: Optional[List[dict]] = None,
        ) -> dict:
            if messages is None:
                return {}

            for message in reversed(messages):
                if message.get("role") == "context":
                    return message
            return {}

        def _get_latest_user_message(
            messages: Optional[List[dict]] = None,
        ) -> dict:
            if messages is None:
                return {}
            for message in reversed(messages):
                if message.get("role") == "user":
                    return message
            return {}

        def _prepare_params(
            flow_id: str,
            action_name: str,
            chunk_str: str,
            prompt: Optional[str] = None,
            messages: Optional[List[dict]] = None,
            action_params: Dict[str, Any] = {},
        ):
            context_message = _get_last_context_message(messages)
            user_message = prompt or _get_latest_user_message(messages)

            context = {
                "user_message": user_message,
                "bot_message": chunk_str,
            }

            if context_message:
                context.update(context_message["content"])

            model_name = flow_id.split("$")[-1].split("=")[-1].strip('"')

            # we pass action params that are defined in the flow
            # caveate, e.g. prmpt_security uses bot_response=$bot_message
            # to resolve replace placeholders in action_params
            for key, value in action_params.items():
                if value == "$bot_message":
                    action_params[key] = chunk_str
                elif value == "$user_message":
                    action_params[key] = user_message

            return {
                # TODO:: are there other context variables that need to be passed?
                # passing events to compute context was not successful
                # context var failed due to different context
                "context": context,
                "llm_task_manager": self.runtime.llm_task_manager,
                "config": self.config,
                "model_name": model_name,
                "llms": self.runtime.registered_action_params.get("llms", {}),
                "llm": self.runtime.registered_action_params.get(
                    f"{action_name}_llm", self.llm
                ),
                **action_params,
            }

        output_rails_streaming_config = self.config.rails.output.streaming
        buffer_strategy = get_buffer_strategy(output_rails_streaming_config)
        output_rails_flows_id = self.config.rails.output.flows
        stream_first = stream_first or output_rails_streaming_config.stream_first
        get_action_details = partial(
            _get_action_details_from_flow_id, flows=self.config.flows
        )

        async for chunk_list, chunk_str_rep in buffer_strategy(streaming_handler):
            chunk_str = " ".join(chunk_list)

            # Check if chunk_str_rep is a JSON string
            # we yield a json error payload in generate_async when
            # streaming has errors
            try:
                json.loads(chunk_str_rep)
                yield chunk_str_rep
                return
            except json.JSONDecodeError:
                pass
            if stream_first:
                words = chunk_str_rep.split()
                if words:
                    yield words[0]
                    for word in words[1:]:
                        yield f" {word}"

            for flow_id in output_rails_flows_id:
                action_name, action_params = get_action_details(flow_id)

                params = _prepare_params(
                    flow_id=flow_id,
                    action_name=action_name,
                    chunk_str=chunk_str,
                    prompt=prompt,
                    messages=messages,
                    action_params=action_params,
                )

                # Execute the action. (Your execute_action returns only the result.)
                result = await self.runtime.action_dispatcher.execute_action(
                    action_name, params
                )
                # Include explain info (whatever _update_explain_info does)
                self.explain_info = self._ensure_explain_info()

                # Retrieve the action function from the dispatcher
                action_func = self.runtime.action_dispatcher.get_action(action_name)

                # Use the mapping to decide if the result indicates blocked content.
                if is_output_blocked(result, action_func):
                    reason = f"Blocked by {flow_id} rails."

                    # return the error as a plain JSON string (not in SSE format)
                    # NOTE: When integrating with the OpenAI Python client, the server code should:
                    # 1. detect this JSON error object in the stream
                    # 2. terminate the stream
                    # 3. format the error following OpenAI's SSE format
                    # the OpenAI client will then properly raise an APIError with this error message

                    error_data = {
                        "error": {
                            "message": reason,
                            "type": "guardrails_violation",
                            "param": flow_id,
                            "code": "content_blocked",
                        }
                    }

                    # return as plain JSON: the server should detect this JSON and convert it to an HTTP error
                    yield json.dumps(error_data)
                    return

            if not stream_first:
                words = chunk_str_rep.split()
                if words:
                    yield words[0]
                    for word in words[1:]:
                        yield f" {word}"


def _get_action_details_from_flow_id(
    flow_id: str,
    flows: List[Union[Dict, Any]],
    prefixes: Optional[List[str]] = None,
) -> Tuple[str, Any]:
    """Get the action name and parameters from the flow id.

    First, try to find an exact match.
    If not found, then if the provided flow_id starts with one of the special prefixes,
    return the first flow whose id starts with that same prefix.
    """

    supported_prefixes = [
        "content safety check output",
        "topic safety check output",
    ]
    if prefixes:
        supported_prefixes.extend(prefixes)

    candidate_flow = None

    for flow in flows:
        # If exact match, use it
        if flow["id"] == flow_id:
            candidate_flow = flow
            break

        # If no exact match, check if both the provided flow_id and this flow's id share a special prefix
        for prefix in supported_prefixes:
            if flow_id.startswith(prefix) and flow["id"].startswith(prefix):
                candidate_flow = flow
                # We don't break immediately here because an exact match would have been preferred,
                # but since we're in the else branch it's fine to choose the first matching candidate.
                # TODO:we should avoid having multiple matchin prefixes
                break

        if candidate_flow is not None:
            break

    if candidate_flow is None:
        raise ValueError(f"No action found for flow_id: {flow_id}")

    # we have identified a candidate, look for the run_action element.
    for element in candidate_flow["elements"]:
        if (
            element["_type"] == "run_action"
            and element["_source_mapping"]["filename"].endswith(".co")
            and "execute" in element["_source_mapping"]["line_text"]
            and "action_name" in element
        ):
            return element["action_name"], element["action_params"]

    raise ValueError(f"No run_action element found for flow_id: {flow_id}")