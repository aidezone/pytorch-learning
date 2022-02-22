import glob
import fiftyone as fo
import json

# 参考文档：https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/index.html#custom-formats


# 定义推理结果文件
prediction_file = "./data/my_animals_inferance.jsonl"

# 读取推理结果，形成一个fiftyone的Sample集合
samples = []
fp = open(prediction_file, "r+")
for line in fp:
    linestr = line.strip()
    if linestr == "":
        continue
    lineObj = json.loads(linestr)

    sample = fo.Sample(filepath=lineObj["image_path"])
    sample["标注结果"] = fo.Classification(label=lineObj["label_name"])
    sample["推理结果"] = fo.Classification(label=lineObj["prediction_name"])

    samples.append(sample)
fp.close()


# 创建一个fiftyone数据集
dataset = fo.Dataset("my-classification-dataset")
dataset.add_samples(samples)

# 运行APP
session = fo.launch_app(dataset)

# 等待服务会话
session.wait()