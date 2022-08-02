class PassManager:
    def __init__(self) -> None:
        self._passes = []

    def add(self, p):
        self._passes.append(p)
        return self

    def execute(self, ir):
        try:
            retval = ir
            for i in range(len(self._passes)):
                retval = self._passes[i](retval)
        except:
            import traceback
            traceback.print_exc()
        return retval
