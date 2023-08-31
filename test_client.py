

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from google.api_core import client_options
from google.generativeai import client


class ClientTests(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        client.default_client_config = {}

    def test_api_key_passed_directly(self):
        client.configure(api_key="AIzA_direct")

        client_opts = client.default_client_config["client_options"]
        self.assertEqual(client_opts.api_key, "AIzA_direct")

    def test_api_key_passed_via_client_options(self):
        client_opts = client_options.ClientOptions(api_key="AIzA_client_opts")
        client.configure(client_options=client_opts)

        client_opts = client.default_client_config["client_options"]
        self.assertEqual(client_opts.api_key, "AIzA_client_opts")

    @mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzA_env"})
    def test_api_key_from_environment(self):
        # Default to API key loaded from environment.
        client.configure()
        client_opts = client.default_client_config["client_options"]
        self.assertEqual(client_opts.api_key, "AIzA_env")

        # But not when a key is provided explicitly.
        client.configure(api_key="AIzA_client")
        client_opts = client.default_client_config["client_options"]
        self.assertEqual(client_opts.api_key, "AIzA_client")

    def test_api_key_cannot_be_set_twice(self):
        client_opts = client_options.ClientOptions(api_key="AIzA_client_opts")

        with self.assertRaisesRegex(ValueError, "You can't set both"):
            client.configure(api_key="AIzA_client", client_options=client_opts)

    def test_api_key_and_client_options(self):
        # Client options should merge with an API key, as long as they are both
        # do not have the key set.
        client_opts = client_options.ClientOptions(api_endpoint="web.site")
        client.configure(api_key="AIzA_client", client_options=client_opts)

        actual_client_opts = client.default_client_config["client_options"]
        self.assertEqual(actual_client_opts.api_key, "AIzA_client")
        self.assertEqual(actual_client_opts.api_endpoint, "web.site")

    @parameterized.parameters(
        client.get_default_discuss_client,
        client.get_default_text_client,
        client.get_default_discuss_async_client,
        client.get_default_model_client,
    )
    @mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "AIzA_env"})
    def test_configureless_client_with_key(self, factory_fn):
        _ = factory_fn()

        # And ensure that it has set the default options.
        actual_client_opts = client.default_client_config["client_options"]
        self.assertEqual(actual_client_opts.api_key, "AIzA_env")


if __name__ == "__main__":
    absltest.main()
