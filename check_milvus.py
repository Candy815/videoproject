from pymilvus import connections, Collection
from config.settings import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION

# 连接Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(MILVUS_COLLECTION)

# 查看统计
print(f"集合名称: {MILVUS_COLLECTION}")
print(f"总向量数: {collection.num_entities}")

# 查询前5条数据
collection.load()
results = collection.query(
    expr="id >= 0",
    output_fields=["id", "video_id", "timestamp"],
    limit=5
)

print("\n前5条数据:")
for r in results:
    print(f"  ID: {r['id']}, 视频: {r['video_id']}, 时间戳: {r['timestamp']}s")