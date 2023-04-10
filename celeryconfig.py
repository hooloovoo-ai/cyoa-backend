broker_url = "amqp://"

result_backend = "rpc://"

result_expires = 3600

task_routes = {
    "backend.generate.generate": {"queue", "generate"},
    "backend.alpaca.alpaca": {"queue", "alpaca"}
}
