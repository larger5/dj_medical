from django.http import JsonResponse

# Create your views here.
def prediction(request, sym):
    print(sym)

    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import SparkSession
    import datetime

    start = datetime.datetime.now()
    # -----------------------------------------------------------------------------------------
    # 0.构建 Spark 对象
    spark = SparkSession.builder.master("local").appName("medical").getOrCreate()

    # 3.训练模型：加载的方式
    loadedPipelineModel = PipelineModel.load('C:\LLLLLLLLLLLLLLLLLLL\BigData_AI\pyspark\model')
    # loadedPipelineModel = PipelineModel.load('/usr/src/app/model')

    # 4.测试数据
    test = spark.createDataFrame([
        (0, sym)
    ], ["id", "symptom"])

    # 5.模型预测 —— 1 妇科疾病、2 神经系统疾病、3 循环系统疾病、4 呼吸系统疾病、5 消化系统疾病
    prediction = loadedPipelineModel.transform(test)
    selected = prediction.select("id", "symptom", "probability", "prediction")
    # https://github.com/apache/spark/blob/master/examples/src/main/python/ml/pipeline_example.py
    for row in selected.collect():
        rid, symptom, prob, predic = row
        print("(%d, %s) --> prob=%s, prediction=%f" % (rid, symptom, str(prob), predic))

    # -----------------------------------------------------------------------------------------
    end = datetime.datetime.now()
    print((end - start).microseconds)  # 551669
    return JsonResponse({'data': predic})
