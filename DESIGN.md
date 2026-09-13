# 02-科研 设计文档：arxiv-sanity-lite 重写（minisanity）

## 参照项目
karpathy/arxiv-sanity-lite（1.7k★）：定时轮询 arXiv API → 建立本地索引 → 用户点赞（vote）→
tf-idf + 线性 SVM 训练个人兴趣分类器 → 按相似度推荐。数据存 pickle，无 Web 框架，前端为静态页。

## 重写范围与增强
1. **存储**：pickle → SQLite（内置 sqlite3，无外部依赖）。
2. **检索/推荐**：保留 tf-idf 余弦相似度；把 SVM 分类器换成"点赞加权质心"推荐
   （无外部 ML 依赖，效果直观可解释），接口上预留 embedding 向量列，将来可无缝换 LLM embedding。
3. **API**：静态页 → FastAPI REST 接口（/papers、/vote、/recommend、/digest）。
4. **arXiv 源**：`ArxivSource` 抽象 + `FixtureSource`（离线测试用），真实源用 urllib 直接调
   arXiv Atom API，不引入额外依赖。
5. **测试**：全离线 fixture 驱动。

## 目录
```
02-research/
├── DESIGN.md
├── minisanity/
│   ├── store.py        SQLite 存取（papers + votes + tfidf 矩阵缓存）
│   ├── sources.py      ArxivSource / FixtureSource
│   ├── recommend.py    tf-idf 索引 + 质心推荐 + digest 生成
│   ├── api.py          FastAPI 应用
│   └── test_minisanity.py
└── README.md
```
