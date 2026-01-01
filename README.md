# 使用说明

```python
# 安装
python -m pip install -e ".[torch]"

# 检查是否已经可以导入本地的diffusers
python -c "import diffusers; print('Version:', diffusers.__version__); print('Path:', diffusers.__file__)"


# 运行 BBuf佬提供的脚本来执行generate

生成的embedding信息在prompt_embeddings这个目录下，如果有多个prompt，只会保存最后一个prompt的embedding，因此建议只用一个prompt测试

有了embedding之后可以让llm参考load_prompt_embeddings.py给sglang的text_encoder模块加一个load脚本，或者直接参考我给的sglang分支
```
