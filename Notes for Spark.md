# Notes for Spark (in Python)

## 0. Environment

- python 2.7.x
- Spark 2.2.0 (may be 2.3.0)
- Hadoop 2.7?

## 1. Getting started

### 1.1 Starting Point: SparkSession

- First, a `SparkSession` shall be created

  ```python
  from pyspark.sql import SparkSession
  
  spark = SparkSession \
      .builder \
      .appName("Python Spark SQL basic example") \
      # set "local[4]" to run locally with 4 cores
      .master("Spark master URL") \
      .config("spark.some.config.option", "some-value") \
      .getOrCreate()  # Gets an existing SparkSession, otherwise creates a new one
  ```

- **Note that** before version `Spark 2.0`, `SparkConf` and `SparkContext` should be created to enable further APIs. However, since version `Spark 2.0`, all these settings are **integrated** into `SparkSession` .

- You can also set the configurations from a `SparkConf` as follows

- ```python
  >>> from pyspark.conf import SparkConf
  >>> SparkSession.builder.config(conf=SparkConf())
  ```

- And you can enable new settings as follows

- ```python
  # set new runtime options
  spark.conf.set("spark.sql.shuffle.partitions", 6)
  spark.conf.set("spark.executor.memory", "2g")
  # get all settings
  spark.conf.getAll()
  ```

- A great example for `.getOrCreate()`

- ```python
  >>> s1 = SparkSession.builder.config("k1", "v1").getOrCreate()
  >>> s1.conf.get("k1") == s1.sparkContext.getConf().get("k1") == "v1"
  True
  # In case an existing SparkSession is returned, the config options specified in this builder will be applied to the existing SparkSession.
  >>> s2 = SparkSession.builder.config("k2", "v2").getOrCreate()
  >>> s1.conf.get("k1") == s2.conf.get("k1")
  True
  >>> s1.conf.get("k2") == s2.conf.get("k2")
  True
  ```

- *[more details](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.SparkSession) about `SparkSession`*

### 1.2 Creating DataFrames

- With a `SparkSession`, applications can create DataFrames from an [existing `RDD`](http://spark.apache.org/docs/latest/sql-programming-guide.html#interoperating-with-rdds), from a Hive table, or from [Spark data sources](http://spark.apache.org/docs/latest/sql-programming-guide.html#data-sources).

- For example, we can load a **CSV** file as follows

  ```python
  # spark is an existing SparkSession
  df = spark.read.load("examples/src/main/resources/people.csv",
                       format="csv", sep=":", inferSchema="true", header="true")
  ```

- A more concrete example, we can load a **text** file as follows

- ```python
  # ./testSpark.txt:
  #                 id\tname\tgrade
  #                 1\tTom\t90
  #                 2\tNamy\t80
  #                 3\tJerry\t50
  >>> df = spark.read.load("./testSpark.txt",
                       format="csv", sep="\t", inferSchema="true", header="true")
  >>> df.show()
  +---+-----+-----+
  | id| name|grade|
  +---+-----+-----+
  |  1|  Tom|   90|
  |  2| Namy|   80|
  |  3|Jerry|   50|
  +---+-----+-----+
  ```

  - get a column by 'name'

  - ```python
    >>> df.grade
    Column<grade>
    ```

  - apply operations like RDD

  - ```python
    >>> df.filter(df.grade > 60).collect()
    [Row(id=1, name=u'Tom', grade=90), Row(id=2, name=u'Namy', grade=80)]
    ```

  - [more details](http://spark.apache.org/docs/2.2.0/api/python/pyspark.sql.html#pyspark.sql.DataFrame) about `DataFrame`



## 2. ML Pipelines

ML Pipelines provide a uniform set of high-level APIs built on top of [DataFrames](http://spark.apache.org/docs/latest/sql-programming-guide.html) that help users create and tune practical machine learning pipelines. The pipeline concept is mostly inspired by the [scikit-learn](http://scikit-learn.org/) project.

- [**DataFrame**](http://spark.apache.org/docs/latest/ml-pipeline.html#dataframe): This ML API uses `DataFrame` from Spark SQL as an ML dataset, which can hold a variety of data types. E.g., a `DataFrame` could have different columns storing text, feature vectors, true labels, and predictions.
- [**Transformer**](http://spark.apache.org/docs/latest/ml-pipeline.html#transformers): A `Transformer` is an algorithm which can transform one `DataFrame` into another `DataFrame`. E.g., an ML model is a `Transformer` which transforms a `DataFrame` with features into a `DataFrame` with predictions.
- [**Estimator**](http://spark.apache.org/docs/latest/ml-pipeline.html#estimators): An `Estimator` is an algorithm which can be fit on a `DataFrame` to produce a `Transformer`. E.g., a learning algorithm is an `Estimator` which trains on a `DataFrame` and produces a model.
- [**Pipeline**](http://spark.apache.org/docs/latest/ml-pipeline.html#pipeline): A `Pipeline` chains multiple `Transformer`s and `Estimator`s together to specify an ML workflow.
- [**Parameter**](http://spark.apache.org/docs/latest/ml-pipeline.html#parameters): All `Transformer`s and `Estimator`s now share a common API for specifying parameters.

### Transformer and Estimator 

Each instance of a `Transformer` or `Estimator` has a unique ID, which is useful in **specifying parameters** (discussed below).

A `Pipeline` is specified as a sequence of stages, and each stage is **either** a `Transformer` or an `Estimator`. These stages are run in order, and the input `DataFrame` is transformed as it passes through each stage. For `Transformer` stages, the `transform()` method is called on the `DataFrame`. For `Estimator` stages, the `fit()` method is called to produce a `Transformer` (which becomes part of the `PipelineModel`, or fitted `Pipeline`), and that `Transformer`’s `transform()` method is called on the `DataFrame`.

[An example](http://spark.apache.org/docs/latest/ml-pipeline.html#how-it-works)

[Details](http://spark.apache.org/docs/latest/ml-pipeline.html#details)

[Set parameters](http://spark.apache.org/docs/latest/ml-pipeline.html#parameters)

## 3. Train a Model

### 3.1 Extract features

- [FeatureHasher](http://spark.apache.org/docs/latest/ml-features.html#featurehasher)

- ```
  
  ```

- 





## References

1. [Spark Official site](http://spark.apache.org/)
2. [Spark 2.0系列之SparkSession详解](https://blog.csdn.net/u013063153/article/details/54615378)