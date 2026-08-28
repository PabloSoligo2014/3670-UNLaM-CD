import requests
import pandas as pd
from datetime import datetime, timedelta
#/demanda-svc/demanda/DiasFeriados
from pathlib import Path
url_base = "https://api.cammesa.com/"


def get_businessday(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    url = url_base + "demanda-svc/demanda/EsDiaHabil"
    df_l = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json"
    }
    try:

        while start_date <= end_date:
            # Formatear a string YYYY-MM-DD
            print(f"Consultando API para la de dias habiles fecha: {start_date.strftime('%Y-%m-%d')}")

            params = {
                "fecha": start_date.strftime("%Y-%m-%d")       
            }

            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            isBusinessDay = response.json()
            # Extraer registros
            df_l.append({"fecha":start_date, "esDiaHabil":isBusinessDay})

            start_date += timedelta(days=1)
        df = pd.DataFrame(df_l)
        #df["fecha"] = pd.to_datetime(df["fecha"])
        return df
    
    except requests.exceptions.RequestException as e:
        detail = ""
        if getattr(e, "response", None) is not None:
            detail = f" - {e.response.text[:500]}"
        print(f"Error en la petición a CAMMESA: {e}{detail}")
        return pd.DataFrame()


    return df

    


def get_holidays(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Obtiene los días feriados desde la API de CAMMESA.
    """
    
    url = url_base + "demanda-svc/demanda/DiasFeriados"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json"
    }

    params = {
        "desde": start_date.strftime("%Y-%m-%d"),
        "hasta": end_date.strftime("%Y-%m-%d")
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Extraer registros
        df = pd.DataFrame(data)

        # Mapeo y limpieza de campos devueltos por la API
        if not df.empty:
           return df
        else:
            print("La API no devolvió datos para el año seleccionado.")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        detail = ""
        if getattr(e, "response", None) is not None:
            detail = f" - {e.response.text[:500]}"
        print(f"Error en la petición a CAMMESA: {e}{detail}")
        return pd.DataFrame()


def get_byRegion(dtime: datetime, region: int) -> pd.DataFrame:
    """
    Obtiene la curva de demanda desde la API de CAMMESA.
    Fechas en formato 'YYYY-MM-DD'.
    """
    url = url_base + "demanda-svc/demanda/ObtieneDemandaYTemperaturaRegionByFecha"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json"
    }


    params = {
        "fecha": dtime.strftime("%Y-%m-%d"),
        "id_region": region
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Extraer registros
        df = pd.DataFrame(data)

        # Mapeo y limpieza de campos devueltos por la API
        # La API suele devolver campos como 'fecha', 'demanda' / 'potencia' y 'temperatura'
        if not df.empty:
            if 'dem' in df.columns and 'demanda' not in df.columns:
                df = df.rename(columns={'dem': 'demanda'})
            # Identificar columnas dinámicamente si los nombres varían ligeramente
            df['fecha'] = pd.to_datetime(df['fecha'])
            df = df.sort_values('fecha').reset_index(drop=True)
            
            # Retornamos solo las variables de interés
            cols_interes = [c for c in ['fecha', 'demanda', 'potencia', 'temperatura', 'temp'] if c in df.columns]
            return df[cols_interes]
        else:
            print("La API no devolvió datos para el rango seleccionado.")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        detail = ""
        if getattr(e, "response", None) is not None:
            detail = f" - {e.response.text[:500]}"
        print(f"Error en la petición a CAMMESA: {e}{detail}")
        return pd.DataFrame()

def get_byRegion_historical(start_date: datetime, end_date: datetime, region: int) -> pd.DataFrame:
    df = pd.DataFrame()
    while start_date <= end_date:
        # Formatear a string YYYY-MM-DD
        print(f"Consultando API para la fecha: {start_date.strftime('%Y-%m-%d')} y región: {region}")
    
        df_temp = get_byRegion(start_date, region=region)
        df = pd.concat([df, df_temp], ignore_index=True)
        start_date += timedelta(days=1)

    return df
            
    
# Ejemplo de uso:
if __name__ == "__main__":

    start_date      = datetime.strptime("2026-01-01", "%Y-%m-%d")
    end_date        = datetime.strptime("2026-08-10", "%Y-%m-%d")

    #df_holidays     = get_holidays(start_date, end_date)
    #df_holidays.to_csv("cammesa_holidays.csv", index=False)
        

    #df_historical   = get_byRegion_historical(start_date, end_date, region=1002)
    #df_historical.to_csv("cammesa_historical.csv", index=False)

    df_business = get_businessday(start_date, end_date)  
    df_business.to_csv("cammesa_businessday.csv", index=False) 
    