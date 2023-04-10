from celery import Celery

app = Celery("cyoa", broker="amqp://", backend="rpc://")
app.conf.update(result_expires=3600)
