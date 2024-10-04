class FieldsetVariable:
    def __init__(self, name, eval_func):
        self.name = name
        self.eval_func = eval_func

    def value(self, fieldset):
        return self.eval_func(fieldset)

    def __repr__(self):
        return f"<FieldsetVariable {self.name}>"