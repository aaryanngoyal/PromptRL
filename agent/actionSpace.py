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
            self.add_2_shot,
            self.add_3_shot,
            self.add_category_definitions,
            self.add_anti_instruction,
            self.strip_examples,
            self.strip_definitions,
            self.shorten_to_bare
        ]

        # tracking of actions taken place on prompts
        
        self.applied_modification  = []

    def add_shot_example(self, prompt):

        """
        Insert 4-shot example before instruction
        """

        if "Example:" in prompt:
            return prompt

        example = (
            "Example: 'UN holds emergency session on border dispute' -> World\n"
            "Example: 'Olympic committee weighs new sports for 2028 games' -> Sports\n"
            "Example: 'Apple reports record quarterly earnings' -> Business\n"
            "Example: 'Apple's AI chip ambitions face regulatory scrutiny in EU' -> Sci/Tech\n\n"
        )
        modified_prompt = example + prompt
        self.applied_modification.append('shot_example')
        return modified_prompt

    def remove_shot_example(self, prompt):

        """
        Removes shot example if it exists
        """

        if not prompt.startswith("Example:"):
            return prompt

        lines = prompt.split('\n\n', 1)

        if len(lines) < 2:
            return prompt

        if "shot_example" in self.applied_modification:
            self.applied_modification.remove("shot_example")
        return lines[-1]

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
        Return a shorter but still grammatical prompt.
        Doesn't mangle — instead returns a fixed, compressed instruction.
        """

        if prompt.strip().startswith("Classify"):
            return (
                "Classify the news article into one of: "
                "World, Sports, Business, Sci/Tech. Output only the label."
            )
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

    def add_2_shot(self, prompt):

        """
        Insert 2-shot examples in prompt
        """

        if prompt.startswith("Example:") and prompt.count("Example:") >= 2:
            return prompt
        examples = (
            "Example: 'Apple stock rises 5%' -> Business\n"
            "Example: 'Scientists discover new species' -> Sci/Tech\n\n"
        )
        self.applied_modification.append('shot_example')
        return examples + self.remove_shot_example(prompt)

    def add_3_shot(self, prompt):

        """
        Insert 3-shot examples in prompt
        """

        if prompt.startswith("Example:") and prompt.count("Example:") >= 3:
            return prompt
        examples = (
            "Example: 'Apple stock rises 5%' -> Business\n"
            "Example: 'Scientists discover new species' -> Sci/Tech\n"
            "Example: 'Insurgents Attack Iraq Election Offices' -> World\n\n"
        )
        self.applied_modification.append('shot_example')
        return examples + self.remove_shot_example(prompt)

    def add_category_definitions(self, prompt):

        """
        Insert category definitions in prompt
        """

        if "Definitions:" in prompt:
            return prompt

        defs = (
            "Definitions:\n"
            "World = international news, politics, conflicts\n"
            "Sports = athletic events, teams, players\n"
            "Business = companies, markets, earnings\n"
            "Sci/Tech = science, technology, software, research\n\n"
        )

        return defs + prompt

    def add_anti_instruction(self, prompt):

        """
        Insert Anti-Instruction to prompt
        """

        if "Do not explain" in prompt or "Output only" in prompt:
            return prompt
        return prompt + "\nDo not explain. Output only the label"

    def strip_examples(self, prompt):

        """
        Removes examples from prompt
        """

        if not "Example:" in prompt:
            return prompt

        lines = prompt.split("\n\n", 1)
        return lines[-1]

    def strip_definitions(self, prompt):

        """
        Removes definitions from prompt
        """

        if not "Definitions:" in prompt:
            return prompt

        lines = prompt.split("\n\n", 1)
        return lines[-1]

    def shorten_to_bare(self, prompt):

        """
        Reduces a prompt to minimal instruction
        """

        return "Categorize the article into one of the following: World, Sports, Business, Sci/Tech"