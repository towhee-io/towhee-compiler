class PassManager:
    def __init__(self) -> None:
        self._passes = []

    def add(self, p):
        return self._passes.append(p)

    def execute(self, ir):
        for p in self._passes:
            ir = p(ir)
        return ir
