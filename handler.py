import pickle
import pandas as pd
from flask             import Flask, request, Response
from rossmann.Rossmann import Rossmann

# loading model
model = pickle.load( open( "C:\\Users\\Luana\\repos\\DataScience-Em-Producao\\model_rossmann.pkl", "rb" )  )

# initialize API
app = Flask( __name__ )

# Precisamos criar um endpoint, uma url que vai receber dados, requests... (usando método route do app)
# metodo pode ser POST ou GET aqui
@app.route( "/rossmann/predict", methods=["POST"] )
def rossman_predict():
    test_json = request.get_json()
    
    # se tivermos dados:
    if test_json:
        # pega o json e converte em df
        # assim abaixo funcionara para 1 linha
        if isinstance( test_json, dict ):
            test_raw = pd.DataFrame( test_json, index=[0] )
        else:
        # mas para o caso de receber varios jsons concatenados... (json é chave:valor. As chaves vao ser as colunas)
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys())
        
        # Instantiate Rossmann class (quando fazemos isso temos acesso a todos os metodos que implementamos na Classe Rossmann)
        pipeline = Rossmann()
        
        # data cleaning
        df1 = pipeline.data_cleaning( test_raw )
        
        # feature engineering
        df2 = pipeline.feature_engineering( df1 )
        
        # data preparation (para poder fazer a requisição para o modelo)
        df3 = pipeline.data_preparation( df2 )
        
        # prediction        
        # passamos o modelo treinado do xgboost (model), os dados originais (test_raw) pois queremos voltar os dados originais para a pessoa 
        # e não os dados transformados e os dados transformados (df3) que vamos usar para fazer as predições do modelo
        df_response = pipeline.get_prediction( model, test_raw, df3 )
        
        return df_response
   
    # se não tivermos dados:
    else:
        return Response( "{}", status=200,  mimetype="application/json" )

# quando o interpretador do python (Handler.py) encontrar a função main dentro da Classe, chama a flask para o rodar a funçao flask
if __name__ == "__main__":
    # so acessivel localmente, para o localhost
    app.run("0.0.0.0")