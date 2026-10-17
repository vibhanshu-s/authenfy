class AnalysisError(Exception):
    def __init__(self, message, status=422, code="invalid_media"):
        super().__init__(message)
        self.status = status
        self.code = code
