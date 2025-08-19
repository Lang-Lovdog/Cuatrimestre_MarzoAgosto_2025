from lovdog_edas_project_h import ModelSelection
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('./transformed_data.csv')
    categoricalColumn = "target"
    Values = {
      "Graduate": 1,
      "Dropout": 0
    }
    df[categoricalColumn] = df[categoricalColumn].map(Values)
    model=ModelSelection("Seleccón", csv=df)
    model.selectModels()
    model.printSummary()
    model.testModels()
    model.printBestModel()
    model.saveModels("models/")
    model.saveConfusionMatrix()

