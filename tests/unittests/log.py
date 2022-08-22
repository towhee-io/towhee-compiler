from unittest import TestCase


class TestLog(TestCase):
    def test_import_log(self):
        from towhee.compiler.log import get_logger
        log = get_logger("test_import_log")
        with self.assertLogs("test_import_log"):
            log.info("test")
