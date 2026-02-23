import re
import yaml

class PlateValidator:
    def __init__(self, schema_path):
        
        with open(schema_path,"r") as f:
            schema=yaml.safe_load(f)
            
        self.total_length=schema["total_length"]
        self.regex_pattern=re.compile(schema["regex_pattern"])
        self.swap_map=schema["swap_map"]
        
    def validate(self,text):
        
        if text is None:
            return ""
        
        if len(text) != self.total_length :
            return ""
        
        if self.regex_pattern.fullmatch(text):
            return text
        
        corrected=self.__attempt_swap(text)
        
        if corrected and self.regex_pattern.fullmatch(corrected):
            return corrected
        
        return ""
    
    def __attempt_swap(self,text):
        
        chars=list(text)
        
        for i,ch in enumerate(chars):
            if ch in self.swap_map:
                chars[i]=self.swap_map[ch]
                
        return "".join(chars)
        