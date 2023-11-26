-- DimCustomer 表
CREATE TABLE DimCustomer(
    CustomerKey INT PRIMARY KEY,
    GeographyKey INT,
    CustomerAlternateKey NVARCHAR(15),
    Gender NVARCHAR(1),
    BirthDate  date NULL,
    YearlyIncome MONEY,
    TotalChildren TINYINT,
    -- 其他字段
);

-- DimDate 表
CREATE TABLE DimDate(
    DateKey INT PRIMARY KEY,
    FullDateAlternateKey DATE,
    EnglishDayNameOfWeek NVARCHAR(10),
    MonthNumberOfYear TINYINT,
    CalendarYear SMALLINT,
    -- 其他字段
);

-- DimGeography 表
CREATE TABLE DimGeography(
    GeographyKey INT PRIMARY KEY,
    City NVARCHAR(30),
    CountryRegionCode NVARCHAR(3),
    -- 其他字段
);

-- DimProduct 表
CREATE TABLE DimProduct(
    ProductKey INT PRIMARY KEY,
    ProductSubcategoryKey INT,
    EnglishProductName NVARCHAR(50),
    StandardCost MONEY,
    Color NVARCHAR(15),
    -- 其他字段
);

-- DimProductCategory 表
CREATE TABLE DimProductCategory(
    ProductCategoryKey INT PRIMARY KEY,
    EnglishProductCategoryName NVARCHAR(50),
    -- 其他字段
);

-- DimProductSubcategory 表
CREATE TABLE DimProductSubcategory(
    ProductSubcategoryKey INT PRIMARY KEY,
    ProductCategoryKey INT,
    EnglishProductSubcategoryName NVARCHAR(50),
    -- 其他字段
);

-- DimReseller 表
CREATE TABLE DimReseller(
    ResellerKey INT PRIMARY KEY,
    GeographyKey INT,
    ResellerName NVARCHAR(50),
    BusinessType varchar, --['Specialty Bike Shop','Value Added Reseller','Warehouse']
    -- 其他字段
);

-- DimSalesTerritory 表
CREATE TABLE DimSalesTerritory(
    SalesTerritoryKey INT PRIMARY KEY,
    SalesTerritoryRegion NVARCHAR(50),
    -- 其他字段
);

-- FactResellerSales 表
CREATE TABLE FactResellerSales(
    SalesOrderNumber NVARCHAR(20) PRIMARY KEY,
    ProductKey INT,
    OrderDateKey INT,
    ResellerKey INT,
    SalesAmount MONEY,
    TotalProductCost money,
    OrderQuantity smallint,
    -- 其他字段
);

-- 外鍵關係
ALTER TABLE DimCustomer ADD CONSTRAINT FK_DimCustomer_DimGeography FOREIGN KEY(GeographyKey) REFERENCES DimGeography(GeographyKey);
ALTER TABLE DimProduct ADD CONSTRAINT FK_DimProduct_DimProductSubcategory FOREIGN KEY(ProductSubcategoryKey) REFERENCES DimProductSubcategory(ProductSubcategoryKey);
ALTER TABLE DimProductSubcategory ADD CONSTRAINT FK_DimProductSubcategory_DimProductCategory FOREIGN KEY(ProductCategoryKey) REFERENCES DimProductCategory(ProductCategoryKey);
ALTER TABLE DimReseller ADD CONSTRAINT FK_DimReseller_DimGeography FOREIGN KEY(GeographyKey) REFERENCES DimGeography(GeographyKey);
ALTER TABLE FactResellerSales ADD CONSTRAINT FK_FactResellerSales_DimProduct FOREIGN KEY(ProductKey) REFERENCES DimProduct(ProductKey);
-- 其他外鍵關係
