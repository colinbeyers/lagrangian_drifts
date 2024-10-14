class FieldsetVariable:
    def __init__(self, name, eval_func):
        self.name = name
        self.eval_func = eval_func

    # A method that can be called on a FieldsetVarible to
    # compute the eval function at runtime (after the fieldset
    # object has been created)
    def value(self, fieldset):
        return self.eval_func(fieldset)

    # This method returns the custom string for the 
    # FielsetVariable object and is similarly worded to 
    # other object strings in PARCELS (for debugging and logging)
    def __repr__(self):
        return f"<FieldsetVariable {self.name}>"