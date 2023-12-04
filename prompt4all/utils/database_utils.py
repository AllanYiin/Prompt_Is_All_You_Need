def parse_connection_string(conn_string: str) -> dict:
    """
    解析數據庫連接字串，從中提取用戶名稱、密碼、伺服器名稱、資料庫名稱及其他參數。

    Args:
        conn_string (str): 要解析的數據庫連接字串。

    Returns:
        dict: 包含解析出的連接參數的字典。可能的鍵包括 'username', 'password', 'server_name',
              'database_name' 以及其他任何在連接字串中指定的參數。

    Examples:
        >>> parse_connection_string('mssql+pyodbc://@localhost/AdventureWorksDW2022?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server')
        {'server_name': 'localhost', 'database_name': 'AdventureWorksDW2022', 'trusted_connection': 'yes', 'driver': 'ODBC+Driver+17+for+SQL+Server'}
        >>> parse_connection_string("mssql+pyodbc://allan:passw0rd@127.0.0.1:1433/msdb?driver=SQL+Server+Native+Client+11.0")
        {'username': 'allan', 'password': 'passw0rd', 'server_name': '127.0.0.1:1433', 'database_name': 'msdb', 'driver': 'SQL+Server+Native+Client+11.0'}

    """
    results = {}
    if conn_string:
        # 分割字符串來提取用戶名稱、密碼、伺服器名稱、資料庫名稱和其他參數
        parts = conn_string.split('://')
        credentials_and_server_info = parts[1].split('@')

        # 處理認證信息
        if ':' in credentials_and_server_info[0]:
            username, password = credentials_and_server_info[0].split(':')
            results['user_name'] = username
            results['password'] = password

        server_and_database_info = credentials_and_server_info[1].split('/')
        results['server_name'] = server_and_database_info[0]
        database_and_parameters = server_and_database_info[1].split('?')

        results['database_name'] = database_and_parameters[0]

        # 處理其他參數
        if len(database_and_parameters) > 1:
            parameters = database_and_parameters[1].split('&')
            for p in parameters:
                try:
                    key, value = p.split('=')
                    results[key] = value
                except:
                    pass

    return results
