from celery import Celery

app = Celery("cyoa", broker="amqp://", backend="rpc://")
app.conf.task_routes = {
    "backend.generate.generate": {"queue", "generate"},
    "backend.alpaca.alpaca": {"queue", "alpaca"}
}
app.conf.update(result_expires=3600)
