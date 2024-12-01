from flask import Flask, request, jsonify
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

DATABASE_URL = "postgresql://adminsp:4SxYdhnha3g6uhgndTvD@178.16.142.77:5432/washinton"
engine = create_engine(DATABASE_URL)

model = None
model_trained = False 

@app.route("/")
def read_root():
    return jsonify({"message": "Welcome to the Low Stock Prediction API"})

@app.route("/train_model", methods=["POST"])
def train_model():
    global model, model_trained

    data = request.get_json()
    warehouse_id = data.get("warehouse_id")

    if not warehouse_id:
        return jsonify({"error": "Invalid input. warehouse_id is required."}), 400

    query = text("""
        SELECT product_id, warehouse_id, stock, "Reserved_Stock"
        FROM inventory
        WHERE warehouse_id = :warehouse_id AND stock IS NOT NULL AND "Reserved_Stock" IS NOT NULL
    """)

    try:
        with engine.connect() as connection:
            result = connection.execute(query, {"warehouse_id": warehouse_id})
            data = pd.DataFrame(result.fetchall(), columns=result.keys())

        if data.empty:
            return jsonify({"error": "No data found for the provided warehouse_id."}), 404

        print("Data from database:", data.head())

        data["remaining_stock"] = data["stock"] - data["Reserved_Stock"]

        threshold = 30  # Umbral para definir bajo stock
        data["low_stock"] = (data["remaining_stock"] < threshold).astype(int)

        X = data[["Reserved_Stock", "stock", "warehouse_id"]]
        y = data["low_stock"]

        X = pd.get_dummies(X, columns=["warehouse_id"], drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("Classification Report:", classification_report(y_test, y_pred))

        model_trained = True

        return jsonify({
            "message": "Model trained successfully for warehouse_id: {}".format(warehouse_id),
            "accuracy": round(accuracy, 4),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_low_stock", methods=["POST"])
def predict_low_stock():
    global model, model_trained

    if not model_trained:
        return jsonify({"error": "Model is not trained yet. Train the model first."}), 400

    try:
        data = request.get_json()
        warehouse_id = data.get("warehouse_id")

        if not warehouse_id:
            return jsonify({"error": "Invalid input. warehouse_id is required."}), 400

        query = text("""
            SELECT product_id, warehouse_id, stock, "Reserved_Stock"
            FROM inventory
            WHERE warehouse_id = :warehouse_id AND stock IS NOT NULL AND "Reserved_Stock" IS NOT NULL
        """)

        with engine.connect() as connection:
            result = connection.execute(query, {"warehouse_id": warehouse_id})
            data = pd.DataFrame(result.fetchall(), columns=result.keys())

        if data.empty:
            return jsonify({"error": "No data found for the provided warehouse_id."}), 404

        print("Data from database:", data.head())

        data["remaining_stock"] = data["stock"] - data["Reserved_Stock"]

        threshold = 30  # Ajustar el umbral según necesidad

        data["low_stock"] = (data["remaining_stock"] <= threshold)

        X = data[["Reserved_Stock", "stock", "warehouse_id"]]
        y = data["low_stock"]

        X = pd.get_dummies(X, columns=["warehouse_id"], drop_first=True)

        print("Data after getting dummies:", X)

        predictions = model.predict(X)

        # Añadimos las predicciones al dataframe
        data["low_stock_prediction"] = predictions

        low_stock_data = data[data["low_stock_prediction"] == 1]

        print("Low stock data:", low_stock_data)  # Verificación de los productos con bajo stock

        low_stock_data_sorted = low_stock_data.sort_values(by="remaining_stock", ascending=True)

        print("Sorted low stock data:", low_stock_data_sorted)

        return jsonify({
            "low_stock_products": low_stock_data_sorted[["product_id", "remaining_stock", "low_stock_prediction"]].to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@app.route("/predict_future_sales", methods=["POST"])
def predict_future_sales():
    try:
        query = text("""
            SELECT store_id, sale_date, total_amount
            FROM sale
        """)

        with engine.connect() as connection:
            result = connection.execute(query)
            sales_data = pd.DataFrame(result.fetchall(), columns=result.keys())

        if sales_data.empty:
            return jsonify({"error": "No sales data found."}), 404

        sales_data['sale_date'] = pd.to_datetime(sales_data['sale_date'])
        sales_data['day'] = (sales_data['sale_date'] - sales_data['sale_date'].min()).dt.days

        future_sales_data = []
        today = pd.Timestamp.now()
        days_since_start = (today - sales_data['sale_date'].min()).days
        future_days = np.array([days_since_start + i for i in range(1, 31)]).reshape(-1, 1)

        for store_id in sales_data['store_id'].unique():
            store_data = sales_data[sales_data['store_id'] == store_id]
            X = store_data[['day']]
            y = store_data['total_amount']

            model = LinearRegression()
            model.fit(X, y)

            future_sales_amounts = model.predict(future_days)
            future_sales_amounts = np.maximum(future_sales_amounts, 0)

            avg_ticket_size = store_data['total_amount'].mean()
            approx_sales_count = np.ceil(future_sales_amounts / avg_ticket_size).astype(int)

            future_dates = pd.date_range(start=today, periods=30).tolist()
            store_future_sales_data = pd.DataFrame({
                'store_id': store_id,
                'sale_date': future_dates,
                'predicted_sales_count': approx_sales_count
            })

            future_sales_data.append(store_future_sales_data)

        # Consolidar resultados
        future_sales_data = pd.concat(future_sales_data).reset_index(drop=True)

        return jsonify({
            "future_sales": future_sales_data.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/predict_top_stores", methods=["POST"])
def predict_top_stores():
    try:
        # Reutilizamos la lógica del endpoint predict_future_sales
        query = text("""
            SELECT store_id, sale_date, total_amount
            FROM sale
        """)

        with engine.connect() as connection:
            result = connection.execute(query)
            sales_data = pd.DataFrame(result.fetchall(), columns=result.keys())

        if sales_data.empty:
            return jsonify({"error": "No sales data found."}), 404

        sales_data['sale_date'] = pd.to_datetime(sales_data['sale_date'])
        sales_data['day'] = (sales_data['sale_date'] - sales_data['sale_date'].min()).dt.days

        today = pd.Timestamp.now()
        days_since_start = (today - sales_data['sale_date'].min()).days
        future_days = np.array([days_since_start + i for i in range(1, 31)]).reshape(-1, 1)

        # Diccionario para almacenar los totales de cada tienda
        store_future_totals = {}

        for store_id in sales_data['store_id'].unique():
            store_data = sales_data[sales_data['store_id'] == store_id]
            X = store_data[['day']]
            y = store_data['total_amount']

            # Entrenamos el modelo
            model = LinearRegression()
            model.fit(X, y)

            # Predicción de las ventas futuras
            future_sales_amounts = model.predict(future_days)
            future_sales_amounts = np.maximum(future_sales_amounts, 0)

            # Total futuro para la tienda actual
            store_future_totals[int(store_id)] = float(future_sales_amounts.sum())

        # Ordenamos las tiendas por el total proyectado
        top_stores_sorted = sorted(store_future_totals.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            "top_stores": top_stores_sorted
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route("/predict_product_demand", methods=["POST"])
def predict_product_demand():
    try:
        query = text("""
            SELECT product.name, product.product_id, sale.sale_date, SUM(sale_detail.quantity) AS total_sold
            FROM sale_detail
            JOIN sale ON sale_detail.sale_id = sale.sale_id
            JOIN product ON sale_detail.product_id = product.product_id
            GROUP BY product.name, product.product_id, sale.sale_date
        """)

        with engine.connect() as connection:
            result = connection.execute(query)
            sales_data = pd.DataFrame(result.fetchall(), columns=result.keys())

        if sales_data.empty:
            return jsonify({"error": "No se encontraron datos de ventas."}), 404

        sales_data['sale_date'] = pd.to_datetime(sales_data['sale_date'])
        sales_data.sort_values(by='sale_date', inplace=True)

        predictions = {}

        for product_name in sales_data['name'].unique():
            product_sales = sales_data[sales_data['name'] == product_name]

            # Indexar por fecha y rellenar días faltantes con 0
            product_sales.set_index('sale_date', inplace=True)
            product_sales = product_sales['total_sold'].resample('D').sum().fillna(0)

            # Crear modelo de regresión lineal
            model = LinearRegression()

            # Convertir las fechas en un índice numérico para entrenamiento
            X = np.arange(len(product_sales)).reshape(-1, 1)
            y = product_sales.values
            model.fit(X, y)

            # Predecir para los próximos 30 días
            future_days = np.arange(len(product_sales), len(product_sales) + 30).reshape(-1, 1)
            future_sales = model.predict(future_days)

            # Guardar las predicciones en el diccionario
            predictions[product_name] = {
                "historical_average": product_sales.mean(),
                "predicted_demand": [max(0, round(value, 2)) for value in future_sales]  # Asegurar que no haya valores negativos
            }

        # Devolver las predicciones en formato JSON
        return jsonify({"predicted_demand": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route("/predict_inventory_rotation", methods=["POST"])
def predict_inventory_rotation():
    try:
        query = text("""
            SELECT inventory.product_id, stock, SUM(sale_detail.quantity) AS total_sold
            FROM inventory
            LEFT JOIN sale_detail ON inventory.product_id = sale_detail.product_id
            GROUP BY inventory.product_id, stock
        """)

        with engine.connect() as connection:
            result = connection.execute(query)
            inventory_data = pd.DataFrame(result.fetchall(), columns=result.keys())

        if inventory_data.empty:
            return jsonify({"error": "No inventory or sales data found."}), 404

        inventory_data['rotation_time'] = inventory_data['stock'] / inventory_data['total_sold'].replace(0, 1)

        return jsonify({
            "inventory_rotation": inventory_data[['product_id', 'rotation_time']].to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/recommended_suppliers", methods=["POST"])
def recommended_suppliers():
    """
    Predice los proveedores recomendados para productos con bajo inventario.
    Devuelve un JSON con el producto, nivel actual y los proveedores recomendados.
    """
    try:
        product_inventory_query = text("""
            SELECT 
                product_batch.product_id, 
                product.name AS product_name,  -- Cambio realizado aquí
                SUM(product_batch.quantity) AS total_quantity
            FROM product_batch
            JOIN product ON product_batch.product_id = product.product_id
            WHERE product_batch.status = 'active'
            GROUP BY product_batch.product_id, product.name  -- Cambio realizado aquí
        """)

        product_supplier_query = text("""
            SELECT 
                product_supplier.product_id, 
                supplier.name AS supplier_name  -- Cambio realizado aquí
            FROM product_supplier
            JOIN supplier ON product_supplier.supplier_id = supplier.supplier_id
        """)

        with engine.connect() as connection:
            inventory_result = connection.execute(product_inventory_query)
            supplier_result = connection.execute(product_supplier_query)

            inventory_data = pd.DataFrame(inventory_result.fetchall(), columns=inventory_result.keys())
            supplier_data = pd.DataFrame(supplier_result.fetchall(), columns=supplier_result.keys())

        if inventory_data.empty or supplier_data.empty:
            return jsonify({"error": "No se encontraron datos de inventario o proveedores."}), 404

        low_stock_products = inventory_data[inventory_data['total_quantity'] < 20]

        recommendations = {}
        for _, row in low_stock_products.iterrows():
            product_id = row['product_id']
            product_name = row['product_name']
            total_quantity = row['total_quantity']

            product_suppliers = supplier_data[supplier_data['product_id'] == product_id]
            supplier_names = product_suppliers['supplier_name'].tolist()

            recommendations[product_name] = {
                "current_inventory": total_quantity,
                "recommended_suppliers": supplier_names
            }

        return jsonify({"recommended_suppliers": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)