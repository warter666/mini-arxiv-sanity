# minisanity — arxiv-sanity-lite 重写

参照 [karpathy/arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite)（1.7k★）的重写版：
追踪 arXiv 新论文、点赞、按个人兴趣推荐。

## 与原版的差异
- pickle → SQLite；静态页 → FastAPI REST
- tf-idf + SVM → tf-idf 余弦 + 点赞加权质心（无 ML 依赖，可解释）
- `Paper.embedding` 列已预留，可无缝升级为 LLM embedding 相似度
- arXiv 抓取做了加固：钉死 https 域名、DNS 解析后拒绝非公网地址（防 SSRF/DNS rebinding）、
  禁止重定向、响应大小上限、解析前拒绝 DTD/ENTITY（防 XML 实体扩展）

## 使用
```bash
pip install fastapi uvicorn
uvicorn minisanity.api:app --reload   # in 02-research/
# 或离线测试：
python -m minisanity.test_minisanity
```

API：`POST /refresh`、`GET /papers`、`POST /vote/{id}?up=true`、
`GET /similar/{id}`、`GET /recommend`、`GET /digest`
