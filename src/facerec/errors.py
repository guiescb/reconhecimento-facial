"""Erros tratados da aplicacao (specs 001 a 004). Cada situacao de rejeicao tem sua classe,
com mensagem em portugues pronta para ser mostrada ao usuario."""


class FacerecError(Exception):
    """Base de todos os erros tratados. `photo_index` (1, 2, 3...) indica qual foto do cadastro
    causou o problema; quando definido, aparece no inicio da mensagem ("Foto 3: ...")."""

    default_message = "erro no processamento."

    def __init__(self, message=None):
        super().__init__(message or self.default_message)
        self.message = message or self.default_message
        self.photo_index = None

    def __str__(self):
        if self.photo_index is not None:
            return f"Foto {self.photo_index}: {self.message}"
        return self.message


class NoFaceDetected(FacerecError):
    default_message = "nenhuma face detectada."


class MultipleFaces(FacerecError):
    def __init__(self, n=2):
        self.n = n
        super().__init__(f"{n} faces detectadas; use fotos com uma única face.")


class InadequateImage(FacerecError):
    default_message = "imagem inadequada."


class DuplicateEnrollment(FacerecError):
    default_message = "essa pessoa parece já estar cadastrada."


class PersonAlreadyExists(FacerecError):
    default_message = "já existe uma pessoa cadastrada com esse nome."


class PersonNotFound(FacerecError):
    default_message = "pessoa não encontrada."


class InvalidName(FacerecError):
    default_message = "nome inválido."


class InvalidPhotoCount(FacerecError):
    default_message = "quantidade de fotos inválida."
