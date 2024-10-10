from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Carregar seu modelo previamente treinado
modelo = joblib.load('seu_modelo.pkl')  # Substitua pelo caminho do seu modelo

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Capturando os dados do formulário
    razao_social = request.form.get('razao_social')
    estado = request.form.get('estado')
    municipio = request.form.get('municipio')
    ano = request.form.get('ano')
    tipo_residuo = request.form.get('tipo_residuo')
    metodo_reciclagem = request.form.get('metodo_reciclagem')
    quantidade = request.form.get('quantidade')
    unidade_medida = request.form.get('unidade_medida')
    empresa_destinadora = request.form.get('empresa_destinadora')

    # Preparar os dados para predição (ajuste conforme necessário)
    dados_entrada = [[
        razao_social, estado, municipio, ano, tipo_residuo,
        metodo_reciclagem, quantidade, unidade_medida, empresa_destinadora
    ]]

    # Fazer a predição
    resultado = modelo.predict(dados_entrada)

    # Retornar o resultado
    return jsonify({'resultado': resultado[0]})

if __name__ == '__main__':
    app.run(debug=True)