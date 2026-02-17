# 新增功能说明

## 1. DeepSeek-chat 评测支持

使用 DeepSeek API 作为 judge，替代 OpenAI GPT 模型。

### 新增文件

**`client_configs/deepseek_configs.yaml`**
配置 OpenAI 客户端指向 DeepSeek API（两者接口兼容）。

**`src/alpaca_eval/evaluators_configs/deepseek_chat/configs.yaml`**
评测器配置，使用 Chain-of-Thought 提示模板，通过正则匹配最后输出的 `m` 或 `M` 来判断胜者。

### 使用方法

```bash
export DEEPSEEK_API_KEY="your-api-key"
alpaca_eval --model_outputs your_outputs.json --annotators_config deepseek_chat
```

DeepSeek API Key 申请地址：https://platform.deepseek.com

---

## 2. client_configs 支持环境变量替换

**改动文件：`src/alpaca_eval/utils.py`**

在 `get_all_clients()` 函数中，YAML 加载后会自动将 `${VAR_NAME}` 替换为对应的环境变量值。

**示例：**
```yaml
default:
    - api_key: "${DEEPSEEK_API_KEY}"
      base_url: "https://api.deepseek.com"
```

执行前设置环境变量即可，无需在配置文件中硬编码密钥：
```bash
export DEEPSEEK_API_KEY="sk-xxx"
```

---

## 3. 独立的 vLLM 推理脚本

**新增文件：`generate_outputs.py`**

不依赖 `models_configs` 配置系统，直接指定模型路径生成 AlpacaEval 评测所需的输出文件。

### 使用方法

```bash
# 基本用法
python generate_outputs.py --model /path/to/model --output outputs.json

# 完整参数
python generate_outputs.py \
  --model /path/to/model \
  --output outputs/my_model.json \
  --gpus 2 \
  --batch_size 512 \
  --max_tokens 4096 \
  --temperature 0.7 \
  --top_p 1.0
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必填 | 模型路径或 HuggingFace repo（如 `Qwen/Qwen2.5-7B-Instruct`） |
| `--output` | 必填 | 输出 JSON 文件路径 |
| `--gpus` | 1 | 使用的 GPU 数量（vLLM tensor_parallel_size） |
| `--batch_size` | 256 | vLLM 最大并发序列数（max_num_seqs） |
| `--max_tokens` | 4096 | 每条输出的最大 token 数 |
| `--temperature` | 0.7 | 采样温度 |
| `--top_p` | 1.0 | Top-p 采样 |

脚本自动从 HuggingFace 加载 805 条 AlpacaEval 指令，用 tokenizer 的 `apply_chat_template` 处理 prompt，输出文件格式与 `alpaca_eval` 命令直接兼容。

### 依赖安装

```bash
pip install vllm transformers datasets
```

### 端到端示例

```bash
# 1. 生成模型输出
python generate_outputs.py --model /path/to/my-model --output outputs/my_model.json

# 2. 用 DeepSeek 作为 judge 评测
export DEEPSEEK_API_KEY="sk-xxx"
alpaca_eval --model_outputs outputs/my_model.json --annotators_config deepseek_chat
```
