from .views import stop_django_server

class ErrorMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = None
        try:
            response = self.get_response(request)
        except Exception as e:
            self.handle_exception(request, e)
        return response

    def handle_exception(self, request, exception):
        print(exception)
        return stop_django_server(request)


