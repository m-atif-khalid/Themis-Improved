-- ============================================================
-- COMPAS training data table (from train.csv)
-- ============================================================
CREATE TABLE COMPAS_TRAINING_DATA (
    Compas_ID                       INT AUTO_INCREMENT PRIMARY KEY,
    Is_Male                         BOOLEAN,
    Age                             INT,
    Race                            VARCHAR(50),
    Number_Of_Juvenile_Fellonies    INT,
    Decile_Score                    INT,
    Number_Of_Juvenile_Misdemeanors INT,
    Number_Of_Other_Juvenile_Offenses INT,
    Number_Of_Prior_Offenses        INT,
    Days_Before_Screening_Arrest    INT,
    Is_Recidivous                   BOOLEAN,
    Days_In_Custody                 INT,
    Is_Violent_Recidivous           BOOLEAN,
    Violence_Decile_Score           INT,
    Two_Year_Recidivous             INT
);
