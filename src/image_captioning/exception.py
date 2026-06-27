import sys


class CustomException(Exception):
    """
    Custom Exception class for the project.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        self.error_message = self.get_detailed_error_message(
            error_message,
            error_detail
        )

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):
        """
        Returns a detailed error message including filename and line number.
        """
        _, _, exc_tb = error_detail.exc_info()

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return (
            f"\nError occurred in Python script:\n"
            f"File Name      : {file_name}\n"
            f"Line Number    : {line_number}\n"
            f"Error Message  : {error_message}"
        )

    def __str__(self):
        return self.error_message