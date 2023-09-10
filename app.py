from flask import Flask, render_template, request
from deploy_models import interpreted_model
from phrases_first import substring_search_first
from pharases_second import substring_search_second
import re

app = Flask(__name__)
app.secret_key = "test12435"


@app.route("/", methods=["GET", "POST"])
def welcome_page():
    if request.method == "POST":
        text = request.form.get("queried_text")
        if request.form.get("model") == "interpreted":
            resulting_tuple = interpreted_model(text)
            substrings = [i.replace(r"\r\n", "") for i in substring_search_first(text)]
            if len(resulting_tuple) == 1:
                for substring in substrings:
                    pattern = re.escape(substring)
                    text = re.sub(pattern, f"<mark>{substring}</mark>", text)
                return render_template(
                    "analyse.html", queried_text=text, certainty=resulting_tuple[0][1], type_class=resulting_tuple[0][0]
                )
            else:
                for substring in substrings:
                    pattern = re.escape(substring)
                    text = re.sub(pattern, f"<mark>{substring}</mark>", text)
                return render_template(
                    "analyse_neighbour.html", queried_text=text, certainty=resulting_tuple[0][1],
                    type_class=resulting_tuple[0][0], certainty_neighbour=resulting_tuple[1][1],
                    type_class_neighbour=resulting_tuple[1][0])
        elif request.form.get("model") == "experiment":
            text = request.form.get("queried_text")
            resulting_tuple = interpreted_model(text)
            substrings = [i.replace(r"\r\n", "") for i in substring_search_second(text)]
            if len(resulting_tuple) == 1:
                for substring in substrings:
                    pattern = re.escape(substring)
                    text = re.sub(pattern, f"<mark>{substring}</mark>", text)
                return render_template(
                    "analyse.html", queried_text=text, certainty=resulting_tuple[0][1],
                    type_class=resulting_tuple[0][0]
                )
            else:
                for substring in substrings:
                    pattern = re.escape(substring)
                    text = re.sub(pattern, f"<mark>{substring}</mark>", text)
                return render_template(
                    "analyse_neighbour.html", queried_text=text, certainty=resulting_tuple[0][1],
                    type_class=resulting_tuple[0][0], certainty_neighbour=resulting_tuple[1][1],
                    type_class_neighbour=resulting_tuple[1][0])
    return render_template("start.html")


@app.route("/info")
def info_page():
    return render_template("info.html")


if __name__ == "__main__":
    app.run()
