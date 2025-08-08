"""
Version of the JSON potential used in the experiments. See genlm-control for a more recent version.
"""

import json_stream
import json

from jsonschema import Draft7Validator, ValidationError
from genlm.control.potential.base import Potential
from jsonschema import _types


def is_sequence(checker, instance):
    from collections.abc import Sequence, Mapping

    return isinstance(instance, Sequence) and not isinstance(
        instance, (str, bytes, bytearray, Mapping)
    )


def is_object(checker, instance):
    from json_stream.base import StreamingJSONObject
    from collections.abc import Mapping

    return isinstance(instance, (Mapping, StreamingJSONObject))


# We're using a streaming JSON library that doesn't return proper lists
# and dicts. In theory we could use jsonschema's custom typechecker logic
# here. In practice, this works until it encounters an explicitly specified
# schema type, at which point it creates a new validator that ignores the
# type checker. There is probably a sensible official way to fix this (I hope)
# but I couldn't figure it out and this was expedient and probably won't
# cause too many problems (I hope) - DRMacIver.
_types.is_array.__code__ = is_sequence.__code__
_types.is_object.__code__ = is_object.__code__


# Ideally we would be using Draft202012Validator for compatibility with
# jsonschemabench, but something about the way it's written makes it worse
# at lazy validation, so we're using an older draft for now.
LazyCompatibleValidator = Draft7Validator


class OutOfBytes(Exception):
    pass


class JustOneBlockIterable:
    """Provides a single value (intended to be bytes from a context)
    and then signals if the reader tried to read past it. This allows
    us to distinguish invalid JSON from incomplete JSON by seeing if
    the reader tried to read more than it had or failed early."""

    def __init__(self, block):
        self.__block = block
        self.read_past_first_block = False

    def __iter__(self):
        yield self.__block
        self.read_past_first_block = True


UTF8_START_BYTE_MASKS = [
    (0b00000000, 0b10000000),
    (0b11000000, 0b11100000),
    (0b11100000, 0b11110000),
    (0b11110000, 0b11111000),
]


def is_utf8_start_byte(n: int) -> bool:
    """Checks if this is a byte that can appear at the
    start of a UTF-8 character."""
    assert 0 <= n < 256
    for prefix, mask in UTF8_START_BYTE_MASKS:
        if n & mask == prefix:
            return True
    return False


class JsonSchema(Potential):
    def __init__(self, schema):
        super().__init__(
            list(range(256)),
        )
        self.schema = schema
        self.validator = LazyCompatibleValidator(
            self.schema, format_checker=Draft7Validator.FORMAT_CHECKER
        )

    def __check_context(self, context):
        context = bytes(context)

        if b"\n" in context:
            raise ValueError("Newline in context")

        # JSON documents have to be valid UTF-8, but we might be
        # in the middle of generating a UTF-8 character. If so, we
        # only consider the prefix that is valid UTF-8, but need
        # to signal at the end that this is a valid prefix and not
        # a valid complete document.
        incomplete_utf8_at_end = False
        try:
            try:
                context.decode("utf-8")
            except UnicodeDecodeError:
                for i in range(1, min(5, len(context))):
                    if is_utf8_start_byte(context[-i]):
                        context = context[:-i]
                        context.decode("utf-8")
                        incomplete_utf8_at_end = True
                        break
                else:
                    raise
        except UnicodeDecodeError:
            raise ValueError("Invalid UTF-8")

        # Feeding just whitespace to json-stream causes it to raise
        # StopIteration, and this is always a valid start to a JSON
        # document of any schema, and never a valid JSON value.
        if not context.strip():
            raise OutOfBytes()

        iterable = JustOneBlockIterable(context)
        try:
            x = json_stream.load(iterable, persistent=True)
            self.validator.validate(x)
            if hasattr(x, "read_all"):
                x.read_all()
        except ValueError:
            if iterable.read_past_first_block:
                raise OutOfBytes()
            else:
                raise
        if incomplete_utf8_at_end:
            raise OutOfBytes()

        # json-stream will just read a JSON object off the start of
        # the stream and then stop, so we reparse the whole string
        # with the normal JSON parser to validate it at the end, or
        # we will allow JSON values to be followed by arbitrary nonsense.
        # This should only fire when we'd be
        try:
            json.loads(context)
        except json.JSONDecodeError as e:
            raise ValueError(*e.args)

    async def complete(self, context) -> float:
        # TODO:
        # 1. Create some sort of caching for the validator, so
        #    we can reuse ones from previous calls.
        # 2. Use a Lark JSON grammar as a prefilter to rule out any
        #    bytes that can't be included next in valid JSON.

        try:
            self.__check_context(context)
        except (ValueError, ValidationError, OutOfBytes):
            return -float("inf")

        return 0.0

    async def prefix(self, context) -> float:
        # TODO:
        # 1. Create some sort of caching for the validator, so
        #    we can reuse ones from previous calls.
        # 2. Use a Lark JSON grammar as a prefilter to rule out any
        #    bytes that can't be included next in valid JSON.
        try:
            self.__check_context(context)
        except (ValueError, ValidationError):
            return -float("inf")
        except OutOfBytes:
            pass

        return 0.0


def few_shots_messages_formatter(task: str, schema: dict, system_prompt: str):
    examples = [value for key, value in EXAMPLES_FOR_TASK.items() if task in key]
    messages = [{"role": "system", "content": system_prompt}]
    for task_examples in examples:
        for input, output in task_examples:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})
    messages.append({"role": "user", "content": json.dumps(schema)})
    return messages


DEFAULT_SYSTEM_PROMPT = "You need to generate a JSON object that matches the schema below. Output the JSON object on a single line. DO NOT use multiple lines and DO NOT output any other text."

# Examples taken from JsonSchemaBench dataset.

EXAMPLES_FOR_TASK = {
    ("Snowplow",): [
        (
            '{\n    "additionalProperties": false,\n    "description": "Schema for a JSON Paths file for loading Redshift from JSON or Avro, http://docs.aws.amazon.com/redshift/latest/dg/copy-parameters-data-format.html#copy-json-jsonpaths",\n    "properties": {\n        "jsonpaths": {\n            "items": {\n                "type": "string"\n            },\n            "minItems": 1,\n            "type": "array"\n        }\n    },\n    "required": [\n        "jsonpaths"\n    ],\n    "self": {\n        "format": "jsonschema",\n        "name": "jsonpaths_file",\n        "vendor": "com.amazon.aws.redshift",\n        "version": "1-0-0"\n    },\n    "type": "object"\n}',
            '{"jsonpaths": ["$.user.id", "$.user.name", "$.user.address.street"]}',
        ),
        (
            '{\n    "additionalProperties": false,\n    "description": "Schema for a Google Analytics enhanced e-commerce product impression custom metric entity",\n    "properties": {\n        "customMetricIndex": {\n            "maximum": 200,\n            "minimum": 1,\n            "type": "integer"\n        },\n        "listIndex": {\n            "maximum": 200,\n            "minimum": 1,\n            "type": "integer"\n        },\n        "productIndex": {\n            "maximum": 200,\n            "minimum": 1,\n            "type": "integer"\n        },\n        "value": {\n            "type": [\n                "integer",\n                "null"\n            ]\n        }\n    },\n    "self": {\n        "format": "jsonschema",\n        "name": "product_impression_custom_metric",\n        "vendor": "com.google.analytics.measurement-protocol",\n        "version": "1-0-0"\n    },\n    "type": "object"\n}',
            '{"customMetricIndex": 120, "listIndex": 45, "productIndex": 10, "value": 300}',
        ),
    ],
    ("Github_easy", "Github_hard", "Github_medium", "Github_trivial", "Github_ultra"): [
        (
            '{\n    "$schema": "http://json-schema.org/draft-04/schema#",\n    "definitions": {\n        "address1": {"type": "string"},\n        "address2": {"type": "string"},\n        "city": {"type": "string"},\n        "country": {"type": "string"},\n        "postalCode": {"type": "string"},\n        "state": {"type": "string"}\n    },\n    "description": "A simple address schema",\n    "properties": {\n        "address1": {"$ref": "#/definitions/address1"},\n        "address2": {"$ref": "#/definitions/address2"},\n        "city": {"$ref": "#/definitions/city"},\n        "country": {"$ref": "#/definitions/country"},\n        "postalCode": {"$ref": "#/definitions/postalCode"},\n        "state": {"$ref": "#/definitions/state"}\n    },\n    "type": "object"\n}',
            '{"address1": "123 Main Street", "address2": "Apt 4B", "city": "Seattle", "country": "USA", "postalCode": "98101", "state": "WA"}',
        ),
        (
            '{\n    "$schema": "http://json-schema.org/draft-06/schema#",\n    "definitions": {\n        "ElementType": {\n            "enum": ["component", "directive"],\n            "type": "string"\n        },\n        "SelectorChange": {\n            "properties": {\n                "remove": {\n                    "description": "Remove directive/component",\n                    "type": "boolean"\n                },\n                "replaceWith": {\n                    "description": "Replace original selector with new one",\n                    "type": "string"\n                },\n                "selector": {\n                    "description": "Original selector to apply change to",\n                    "type": "string"\n                },\n                "type": {\n                    "$ref": "#/definitions/ElementType",\n                    "description": "Type of selector the change applies to - either component or directive"\n                }\n            },\n            "required": ["selector", "type"],\n            "type": "object"\n        }\n    },\n    "properties": {\n        "changes": {\n            "description": "An array of changes to component/directive selectors",\n            "items": {\n                "$ref": "#/definitions/SelectorChange"\n            },\n            "type": "array"\n        }\n    },\n    "required": ["changes"],\n    "type": "object"\n}',
            '{\n  "changes": [\n    {\n      "selector": "app-root",\n      "type": "component",\n      "remove": false,\n      "replaceWith": "new-root"\n    },\n    {\n      "selector": "my-directive",\n      "type": "directive",\n      "remove": true,\n      "replaceWith": "new-directive"\n    }\n  ]\n}',
        ),
    ],
    ("Glaiveai2K",): [
        (
            '{"properties": {"username": {"description": "The user\'s username", "type": "string"}, "email": {"description": "The user\'s email address", "type": "string"}, "age": {"description": "The user\'s age", "type": "integer"}, "is_active": {"description": "Whether the user is active", "type": "boolean"}}, "required": ["username", "email"], "type": "object"}',
            '{"username": "johndoe", "email": "john@example.com", "age": 30, "is_active": true} ',
        ),
        (
            '{"properties": {"product_id": {"description": "The ID of the product", "type": "string"}, "rating": {"description": "The rating given by the user", "type": "integer"}, "comments": {"description": "Additional comments about the product", "type": "string"}}, "required": ["product_id", "rating"], "type": "object"}',
            '{"product_id": "12345", "rating": 5, "comments": "Excellent product! Highly recommend."} ',
        ),
    ],
    ("JsonSchemaStore",): [
        (
            '{\n  "$id": "https://json.schemastore.org/minecraft-trim-pattern.json",\n  "$schema": "http://json-schema.org/draft-07/schema#",\n  "description": "A trim pattern for a Minecraft data pack config schema",\n  "properties": {\n    "asset_id": {\n      "type": "string"\n    },\n    "description": {\n      "properties": {\n        "color": {\n          "type": "string"\n        },\n        "translate": {\n          "type": "string"\n        }\n      },\n      "required": ["translate"],\n      "type": "object"\n    },\n    "template_item": {\n      "type": "string"\n    }\n  },\n  "required": ["asset_id", "description", "template_item"],\n  "title": "Minecraft Data Pack Trim Pattern",\n  "type": "object"\n}',
            '{\n  "asset_id": "minecraft:trim_pattern",\n  "description": {\n    "color": "#FFAA00",\n    "translate": "trim_pattern.description"\n  },\n  "template_item": "minecraft:template_item"\n}',
        ),
        (
            '{\n  "$comment": "https://minecraft.fandom.com/wiki/Data_Pack",\n  "$id": "https://json.schemastore.org/minecraft-damage-type.json",\n  "$schema": "http://json-schema.org/draft-07/schema#",\n  "description": "A damage type\'s for a Minecraft data pack config schema",\n  "properties": {\n    "death_message_type": {\n      "enum": ["default", "fall_variants", "intentional_game_design"],\n      "type": "string"\n    },\n    "effects": {\n      "enum": ["hurt", "thorns", "drowning", "burning", "poking", "freezing"],\n      "type": "string"\n    },\n    "exhaustion": {\n      "type": "number"\n    },\n    "message_id": {\n      "type": "string"\n    },\n    "scaling": {\n      "enum": ["never", "always", "when_caused_by_living_non_player"],\n      "type": "string"\n    }\n  },\n  "required": ["message_id", "scaling", "exhaustion"],\n  "title": "Minecraft Data Pack Damage Type",\n  "type": "object"\n}',
            '{\n  "message_id": "minecraft:damage.message",\n  "scaling": "always",\n  "exhaustion": 0.3,\n  "death_message_type": "default",\n  "effects": "hurt"\n}',
        ),
    ],
    ("Kubernetes",): [
        (
            '{\n  "description": "A topology selector requirement is a selector that matches given label. This is an alpha feature and may change in the future.",\n  "properties": {\n    "key": {\n      "description": "The label key that the selector applies to.",\n      "type": ["string", "null"]\n    },\n    "values": {\n      "description": "An array of string values. One value must match the label to be selected. Each entry in Values is ORed.",\n      "items": {\n        "type": ["string", "null"]\n      },\n      "type": ["array", "null"]\n    }\n  },\n  "required": ["key", "values"],\n  "type": "object"\n}',
            '{\n  "key": "region",\n  "values": ["us-west-1", "us-east-1"]\n}',
        ),
        (
            '{\n  "description": "HostAlias holds the mapping between IP and hostnames that will be injected as an entry in the pod\'s hosts file.",\n  "properties": {\n    "hostnames": {\n      "description": "Hostnames for the above IP address.",\n      "items": {\n        "type": ["string", "null"]\n      },\n      "type": ["array", "null"]\n    },\n    "ip": {\n      "description": "IP address of the host file entry.",\n      "type": ["string", "null"]\n    }\n  },\n  "type": "object"\n}',
            '{\n  "ip": "192.168.1.1",\n  "hostnames": ["example.com", "test.com"]\n}',
        ),
    ],
    ("WashingtonPost",): [
        (
            '{\n  "additionalProperties": false,\n  "description": "Models a auxiliary used in targeting a piece of content.",\n  "properties": {\n    "_id": {\n      "description": "The unique identifier for this auxiliary.",\n      "type": "string"\n    },\n    "name": {\n      "description": "The general name for this auxiliary.",\n      "type": "string"\n    },\n    "uid": {\n      "description": "A short identifier for this auxiliary. Usually used in cases where a long form id cannot work.",\n      "type": "string"\n    }\n  },\n  "required": ["_id", "uid"],\n  "title": "Auxiliary",\n  "type": "object"\n}',
            '{\n  "_id": "12345",\n  "uid": "aux123",\n  "name": "Sample Auxiliary"\n}',
        ),
        (
            '{\n  "additionalProperties": {},\n  "definitions": {\n    "trait_additional_properties_json": {\n      "$schema": "http://json-schema.org/draft-04/schema#",\n      "additionalProperties": {},\n      "description": "A grab-bag object for non-validatable data.",\n      "title": "Has additional properties",\n      "type": "object"\n    }\n  },\n  "description": "Comment configuration data",\n  "properties": {\n    "additional_properties": {\n      "$ref": "#/definitions/trait_additional_properties_json"\n    },\n    "allow_comments": {\n      "description": "If false, commenting is disabled on this content.",\n      "type": "boolean"\n    },\n    "comments_period": {\n      "description": "How long (in days) after publish date until comments are closed.",\n      "type": "integer"\n    },\n    "display_comments": {\n      "description": "If false, do not render comments on this content.",\n      "type": "boolean"\n    },\n    "moderation_required": {\n      "description": "If true, comments must be moderator-approved before being displayed.",\n      "type": "boolean"\n    }\n  },\n  "title": "Comments",\n  "type": "object"\n}',
            '{\n  "allow_comments": true,\n  "comments_period": 30,\n  "display_comments": true,\n  "moderation_required": false,\n  "additional_properties": {}\n}',
        ),
    ],
    ("default",): [],
}


def prompt_formatter(
    tokenizer,
    instance,
    use_chat_format=True,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
):
    """Default prompt formatter for JSON Schema.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
        instance (JSONSchemaInstance): The instance to format.
        use_chat_format (bool): Whether to use chat format.
        system_prompt (str): The system prompt to use.

    Returns:
        (list[int]): The prompt ids.
    """
    if use_chat_format:
        return tokenizer.apply_chat_template(
            conversation=few_shots_messages_formatter(
                task=instance.task,
                schema=instance.json_schema,
                system_prompt=system_prompt,
            ),
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        raise NotImplementedError("JSON schema does not support non-chat format")
