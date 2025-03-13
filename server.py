from flask import Flask
from flask import render_template

app = Flask(__name__)

rules = [
        {'id':1,'name': '규칙 1', 'profit': 250, 'yield': 7.5},
        {'id':2,'name': '규칙 2', 'profit': -150, 'yield': -4.2},
        {'id':3,'name': '규칙 3', 'profit': 100, 'yield': 3.0}
    ]

@app.route('/')
def index():
    
    return render_template('index.html', rules=rules)


@app.route('/rule/<int:rule_id>')
def rule_detail(rule_id):
    rule = next((r for r in rules if r['id'] == rule_id), None)
    return render_template('rule_detail.html', rule=rule)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host="0.0.0.0", port=port)


