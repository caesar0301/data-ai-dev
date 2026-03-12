#!/usr/bin/env python3
"""
Daft on Ray - 简单的作业提交示例
"""

import ray
import daft


def main():
    # 连接到 Ray 集群
    print("连接到 Ray 集群...")
    ray.init(
        address="ray://127.0.0.1:10001",
        runtime_env={"pip": ["daft"]},
        ignore_reinit_error=True
    )
    print(f"Ray 集群连接成功: {ray.cluster_resources()}")

    # 显式告诉 Daft 使用 Ray 执行
    print("设置 Daft 使用 Ray 执行...")
    daft.set_runner_ray("ray://127.0.0.1:10001")

    # 创建 Daft DataFrame
    print("\n创建 Daft DataFrame...")
    df = daft.from_pydict({
        "a": [3, 2, 5, 6, 1, 4],
        "b": [True, False, False, True, True, False]
    })

    # 执行 Daft 操作（这些操作会在 Ray 集群上执行）
    print("执行 Daft 操作...")
    result = df.where(df["b"]).sort(df["a"]).collect()

    # 显示结果
    print("\n查询结果:")
    print(result)

    # 添加一些计算
    print("\n执行更复杂的计算...")
    from daft import col
    from daft.functions import when
    df_with_calc = df.with_column(
        "c",
        when(col("b"), then=df["a"] * 2).otherwise(0)
    )
    result_calc = df_with_calc.collect()
    print(result_calc)

    # 关闭 Ray 连接
    print("\n关闭 Ray 连接...")
    ray.shutdown()
    print("作业完成!")


if __name__ == "__main__":
    main()

