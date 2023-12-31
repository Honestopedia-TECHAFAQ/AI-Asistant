import sys
import unittest

if sys.version_info < (3, 11):
    import asynctest
    from asynctest import mock as async_mock

import google.ai.generativelanguage as glm

from google.generativeai import discuss
from absl.testing import absltest
from absl.testing import parameterized

bases = (parameterized.TestCase,)

if sys.version_info < (3, 11):
    bases = bases + (asynctest.TestCase,)

unittest.skipIf(
    sys.version_info >= (3, 11), "asynctest is not suported on python 3.11+"
)


class AsyncTests(*bases):
    if sys.version_info < (3, 11):

        async def test_chat_async(self):
            client = async_mock.MagicMock()

            observed_request = None

            async def fake_generate_message(
                request: glm.GenerateMessageRequest,
            ) -> glm.GenerateMessageResponse:
                nonlocal observed_request
                observed_request = request
                return glm.GenerateMessageResponse(
                    candidates=[
                        glm.Message(
                            author="1", content="Why did the chicken cross the road?"
                        )
                    ]
                )

            client.generate_message = fake_generate_message

            observed_response = await discuss.chat_async(
                model="models/bard",
                context="Example Prompt",
                examples=[["Example from human", "Example response from AI"]],
                messages=["Tell me a joke"],
                temperature=0.75,
                candidate_count=1,
                client=client,
            )

            self.assertEqual(
                observed_request,
                glm.GenerateMessageRequest(
                    model="models/bard",
                    prompt=glm.MessagePrompt(
                        context="Example Prompt",
                        examples=[
                            glm.Example(
                                input=glm.Message(content="Example from human"),
                                output=glm.Message(content="Example response from AI"),
                            )
                        ],
                        messages=[glm.Message(author="0", content="Tell me a joke")],
                    ),
                    temperature=0.75,
                    candidate_count=1,
                ),
            )
            self.assertEqual(
                observed_response.candidates,
                [{"author": "1", "content": "Why did the chicken cross the road?"}],
            )


if __name__ == "__main__":
    absltest.main()
