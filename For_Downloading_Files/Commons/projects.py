from Base.Singleton import Singleton
from Commons.database import get_questions

class LabellerrProject(Singleton):

    def get_questions(self, project_id: str):
        return get_questions(project_id)
    

if __name__ == "__main__":
    print(LabellerrProject.get_instance().get_questions("violante_dull_aardwolf_38200"))