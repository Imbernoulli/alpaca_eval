# DeepSeek Chat Evaluator

This evaluator uses DeepSeek's API (https://api.deepseek.com) which is compatible with OpenAI's API format.

## Setup

1. Get your DeepSeek API key from https://platform.deepseek.com/

2. Set your API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-deepseek-api-key-here"
   ```

   Note: We use `OPENAI_API_KEY` because the OpenAI client automatically reads from this variable.
   Alternatively, you can set the path to the DeepSeek config:
   ```bash
   export DEEPSEEK_CLIENT_CONFIG_PATH="client_configs/deepseek_configs.yaml"
   ```

3. Run AlpacaEval with the DeepSeek evaluator:
   ```bash
   alpaca_eval --model_outputs <your_outputs.json> \
               --annotators_config deepseek_chat
   ```

## Model Information

- Model: `deepseek-chat` (DeepSeek-V3.2 with 128K context limit)
- Base URL: `https://api.deepseek.com`
- This is the non-thinking mode of DeepSeek-V3.2

For the thinking mode (reasoning), use `deepseek-reasoner` instead.
