class ActionSpace:

    """
    Defines the actions which can be performed on the prompts
    """

    def __init__(self):

        # defines the actions 

        self.actions = [
            self.add_shot_example,
            self.remove_shot_example,
            self.add_format_clarifier,
            self.remove_format_clarifier,
            self.simplify_language,
            self.rephrase_instruction,
            self.remove_punctuation,
            self.restore_punctuation
        ]

        # tracking of actions taken place on prompts
        
        self.applied_modification  = []

    def add_shot_example(self, prompt):

        """
        Insert 1-shot example before instruction
        """

        example = "Example: 'Apple stock rises 5%' -> Business \n\n"
        modified_prompt = example + prompt
        self.applied_modification.append('shot_example')
        return modified_prompt

    def remove_shot_example(self, prompt):

        """
        Removes 1-shot example if it exists
        """

        lines = prompt.split('\n\n')
        if lines and lines[0].startswith("Example:"):
            modified_prompt = '\n\n'.join(lines[1:])

            if 'shot_example' in self.applied_modification:
                self.applied_modification.remove('shot_example')
            return modified_prompt
        return prompt

    def add_format_clarifier(self, prompt):

        """
        Add format instruction at the end
        """

        clarifier = "Answer with only one word. Do not use punctuation."
        modified_prompt = prompt + clarifier
        self.applied_modification.append('format_clarifier')
        return modified_prompt

    def remove_format_clarifier(self, prompt):

        """
        Remove format clarifier
        """

        for marker in ["Answer with only one word", "Do not use punctuation"]:
            prompt = prompt.replace(marker, "")
        prompt = " ".join(prompt.replace("  ", " ").split())

        if "format_clarifier" in self.applied_modification:
            self.applied_modification.remove('format_clarifier')
        return prompt

    def simplify_language(self, prompt):

        """
        Shorten the instruction
        """

        keywords = ['Classify', 'Categorize', 'topic', 'category', 'following']
        simplified = " ".join([word for word in prompt.split()
                              if any(key in word for key in keywords)])
        if simplified:
            
            modified_prompt = simplified + " the article"
            self.applied_modification.append("simplified")
            return modified_prompt        
        return prompt

    def rephrase_instruction(self, prompt):

        """
        Change instruction verb
        """


        rephrases = [
            ('Classify', 'Categorize'),
            ('Categorize', 'Identify the topic of'),
            ('Identify the topic of', 'Classify')
        ]

        for old, new in rephrases:
            if old in prompt:
                modified_prompt = prompt.replace(old, new)
                self.applied_modification.append('rephrase')
                return modified_prompt
        return prompt

    def remove_punctuation(self, prompt):

        """
        Remove punctuation from instruction
        """

        import string

        translator = str.maketrans('', '', string.punctuation)
        modified_prompt = prompt.translate(translator)
        self.applied_modification.append('removed_punctuation')
        return modified_prompt

    def restore_punctuation(self, prompt):

        """
        Restore punctuation if removed
        """

        if 'removed_punctuation' in self.applied_modification:
            self.applied_modification.remove('removed_punctuation')
        return prompt