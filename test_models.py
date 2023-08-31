from unittest import mock

from absl.testing import absltest

import google.ai.generativelanguage as glm
from google.ai.generativelanguage_v1beta2.types import model

from google.generativeai import models

_FAKE_MODEL = model.Model(
    name="models/fake-model-001",
    base_model_id="",
    version="001",
    display_name="Fake Model",
    description="A fake model",
    input_token_limit=123,
    output_token_limit=234,
    supported_generation_methods=[],
)


class UnitTests(absltest.TestCase):
    def test_model_prefix(self):
        """Test `models/` prefix applies to get_model calls when necessary."""
        # The SUT needs a concrete return type from `get_model`, so set up a real-enough client.
        fake_client = mock.Mock(spec=glm.ModelServiceClient)
        fake_client.get_model.return_value = _FAKE_MODEL

        # Ensure that we don't mess with correctly structure args.
        models.get_model(name="models/text-bison-001", client=fake_client)
        fake_client.get_model.assert_called_with(name="models/text-bison-001")

        # Ensure that we do correct bare models.
        models.get_model(name="text-bison-001", client=fake_client)
        fake_client.get_model.assert_called_with(name="models/text-bison-001")

        # And unknown structure is not touched.
        models.get_model(name="unknown_string", client=fake_client)
        fake_client.get_model.assert_called_with(name="unknown_string")
