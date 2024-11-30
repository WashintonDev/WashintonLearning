from fastapi import FastAPI

app = FastAPI()

@app.get("/plot")
def read_plot():
    return {"message": "This is the plot endpoint"}