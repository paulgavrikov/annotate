import mimetypes
import google
import google.generativeai as genai
import os
import logging
from pathlib import Path


class GeminiModel():

    def __init__(self, model_name):
        super().__init__()

        # check if the gemini api key is set
        if "GEMINI_API_KEY" not in os.environ:
            raise ValueError("GEMINI_API_KEY not set in environment")

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(model_name=model_name)

        self.temperature = 0.4

    def forward(self, prompt: str, image_path: str) -> dict:
        assert Path(image_path).exists(), f"Image path {image_path} does not exist"

        image_parts = [
            {
                "mime_type": mimetypes.MimeTypes().guess_type(image_path)[0],
                "data": Path(image_path).read_bytes(),
            },
        ]
        prompt_parts = [image_parts[0], "\n" + prompt]

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            top_p=0 if self.temperature == 0 else None,
            top_k=1 if self.temperature == 0 else None,
        )
        self.last_generation_config = generation_config

        try:
            response = self.model.generate_content(
                prompt_parts,
                safety_settings=safety_settings,
                generation_config=generation_config,
            )
            return {"response": response.text.strip()}
        except ValueError as e:
            logging.error(f"Error generating response.")
            logging.error(f"Prompt feedback: {response.prompt_feedback}")

            if response.prompt_feedback.block_reason is not None:
                return {"response": "BLOCKED"}
            else:  # cant handle this case
                raise e
        except google.api_core.exceptions.InternalServerError:
            logging.error(f"Internal server error")
            return {"response": None}