import numpy as np
from numpy.typing import NDArray
from typing import TypedDict, Optional, Union
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 类型定义
Dataset = NDArray[np.float64]

class Leaf(TypedDict):
    label: int  # 叶节点标签

class Node(TypedDict):
    feature: int  # 划分特征（从1开始）
    threshold: float  # 划分阈值
    children: list  # 子节点列表 [左子树, 右子树]

def entropy(class_counts: list[int]) -> float:
    """计算熵"""
    total = sum(class_counts)
    probabilities = [count / total for count in class_counts if count > 0]
    return -sum(p * np.log2(p) for p in probabilities)

def split_dataset(dataset: Dataset, feature: int, threshold: float) -> tuple:
    """按特征和阈值分割数据集"""
    left_mask = dataset[:, feature] <= threshold
    right_mask = ~left_mask
    return dataset[left_mask], dataset[right_mask]

def find_best_split(dataset: Dataset) -> tuple[int, float]:
    """寻找最佳分裂特征和阈值"""
    best_gain = -1
    best_feature = -1
    best_threshold = 0.0

    # 计算原始熵
    class_counts = [np.sum(dataset[:, -1] == (i + 1)) for i in range(3)]
    base_entropy = entropy(class_counts)

    for feature in range(dataset.shape[1] - 1):
        values = np.unique(dataset[:, feature])
        if len(values) < 2:
            continue
        thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

        for threshold in thresholds:
            left, right = split_dataset(dataset, feature, threshold)
            if len(left) == 0 or len(right) == 0:
                continue

            # 计算信息增益
            left_counts = [np.sum(left[:, -1] == (i + 1)) for i in range(3)]
            right_counts = [np.sum(right[:, -1] == (i + 1)) for i in range(3)]
            info_gain = base_entropy - (
                (len(left) / len(dataset)) * entropy(left_counts) +
                (len(right) / len(dataset)) * entropy(right_counts)
            )

            if info_gain > best_gain:
                best_gain = info_gain
                best_feature = feature
                best_threshold = threshold

    return best_feature + 1, best_threshold  # 特征索引从1开始

def build_tree(
        dataset: Dataset,
        max_depth: int = 5,
        min_samples_split: int = 10,
        depth: int = 0
) -> Union[Node, Leaf]:
    """递归构建决策树"""
    # 终止条件
    current_classes = dataset[:, -1]
    if (depth >= max_depth or
        len(dataset) < min_samples_split or
        len(np.unique(current_classes)) == 1):
        counts = [np.sum(current_classes == (i + 1)) for i in range(3)]
        return Leaf(label=np.argmax(counts) + 1)

    # 寻找最佳分裂
    feature, threshold = find_best_split(dataset)
    left_data, right_data = split_dataset(dataset, feature - 1, threshold)

    # 递归构建子树
    children = [
        build_tree(left_data, max_depth, min_samples_split, depth + 1),
        build_tree(right_data, max_depth, min_samples_split, depth + 1)
    ]

    return Node(
        feature=feature,
        threshold=round(threshold, 2),
        children=children
    )

def predict(tree: Union[Node, Leaf], sample: NDArray) -> int:
    """预测单个样本"""
    while 'feature' in tree:
        if sample[tree['feature'] - 1] <= tree['threshold']:
            tree = tree['children'][0]
        else:
            tree = tree['children'][1]
    return tree['label']

def evaluate(tree: Union[Node, Leaf], test_data: Dataset) -> dict:
    """评估模型性能"""
    y_true = test_data[:, -1].astype(int)
    y_pred = [predict(tree, row) for row in test_data]

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[1, 2, 3]))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=[1, 2, 3], zero_division=0))

    return {
        'accuracy': np.mean(y_true == y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'report': classification_report(y_true, y_pred, output_dict=True)
    }

def convert_to_sklearn_tree(tree: Union[Node, Leaf], dataset: Dataset) -> DecisionTreeClassifier:
    """将自定义决策树转换为sklearn的DecisionTreeClassifier对象"""
    X = dataset[:, :-1]
    y = dataset[:, -1]
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf

def visualize_tree(tree: Union[Node, Leaf], dataset: Dataset, filename: str = "tree"):
    """可视化决策树"""
    clf = convert_to_sklearn_tree(tree, dataset)
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=[f"Feature {i+1}" for i in range(dataset.shape[1] - 1)], class_names=["Class 1", "Class 2", "Class 3"])
    plt.savefig(f"{filename}.png")
    plt.show()

if __name__ == "__main__":
    # 加载数据（假设数据格式：最后一列为标签1/2/3）
    try:
        train_data = np.loadtxt("traindata.txt")
        test_data = np.loadtxt("testdata.txt")
    except FileNotFoundError:
        print("请确保 traindata.txt 和 testdata.txt 存在于当前目录")
        exit()

    # 训练模型
    tree = build_tree(
        train_data,
        max_depth=3,  # 控制树深度防止过拟合
        min_samples_split=5  # 节点最少样本数
    )

    # 评估模型
    results = evaluate(tree, test_data)
    print(f"\n测试准确度为: {results['accuracy'] * 100:.2f}%")

    # 可视化决策树
    visualize_tree(tree, train_data, "my_decision_tree")
    print("\n决策树已保存为 my_decision_tree.png")