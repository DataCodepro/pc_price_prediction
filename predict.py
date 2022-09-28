import streamlit as st
import pickle
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_model():
    with open('steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor = data["model"]
le_model = data["le_model"]
le_model_1 = data["le_model_1"]
le_processor = data["le_processor"]
def show_predict_page():
    st.title("PC Price Prediction")

    st.write("""### We need some information to predict the price""")
    models=(
        'EliteBook 2560p', 'Razer Blade 15', 'Acer Nitro AN515-55', 'HP Folio 9480m',
    'HP EliteBook 8440p', 'ASUS ZenBook(UX325EA)', 'HP Pavilion ', 'HP Zbook G3',
    'Hp Zbook 15u G6', 'HP 840 G3' ,'DELL XPS 15 9560',
    'Acer Predator helios 300', 'HP EliteBook X360 1030 G2', 'Lenovo Region 5',
    'SAMSUNG 730QCJ', 'DELL XPS 13(9360)', 'HP Probook 445 G6',
    'HP Pro X2 612 G1', 'ASUS G551 JM', 'LENOVO YOGA', 'LENOVO THINKPAD T490',
    'DELL Precision 7530', 'Hp Laptop 15' ,'LENOVO Thinkpad E15',
    'Dell Latitude E3380', 'HP Elite 1012' ,'Dell Inspiron 7415', 'ACER NITRO 5',
    'DELL PRECISION 7530', 'LENOVO THINKPAD P73', 'DELL ALIENWARE M15 R2',
    'HP ELITE BOOK X360 1012 G1', 'HP ELITE BOOK 1012 G2', 'HP PRO BOOK 450 G3',
    'HP Envy x360' ,'Dell Inspiron 5593', 'Hp Elitebook 850 G6',
    'SONT PCG VPCC W21FX', 'HP ELITEBOOK 650 G5', 'Dell Latitude 5400',
    'Lenovo Thinkpad T420i', 'HP Omen 16', 'LENOVE G470',
    'Hp Zbook Firefly 15 G7 ', 'SONY PGV VPCEB29FJ', 'TRANSFORMATION BOOK ',
    'MACBOOK PRO', 'MSi Katana Gf66', 'HP PROBOOK 645' ,'Dell G15 5510', 'HP 215',
    'ULTRA SLIM HP 14', 'DELL INSPIRON 15 ', 'LENOVO X1', 'PAVILION g7-1150us',
    'DELL STUDIO 1555', 'Lenovo Thinkbood 15 G2', 'LENOVO THINKPAD T440P',
    'Microsoft Surface Pro 6', 'DELL LATTITUDE E6410', 'LENOVO IDEAPAD 151WL',
    'HP laptop 14', 'Dell latitude 3500', 'Dell Latitude 7480', 'HP Pavilion 15',
    'HP Elitebook 830 G6', 'HP Elitebook 745 G5', 'Dell G15 5511',
    'Lenovo Thinkpad T14s', 'MACBOOK PRO 2017', 'HP Elitebook 1030 G2',
    'HP probook 11 G2', 'LENOVO YOGA 710', 'HP Notebook 15',
    'HP Elitebook 450 G5', 'Dell Inspiron 5567', 'Lenovo Ideapad 5',
    'Dell Inspiron 5579', 'SAMSUNG A5A5' ,'MSi Pulse GL66 11UCK',
    'Lenovo Ideapad S340', 'Dell Inspiron 7391`', 'Dell Latitude 7400',
    'ASUS GL502VMK', 'MSI Creator 15 A10SET', 'Asus Vivo book17',
    'Dell Vostro 5490', 'Asus Notebook', 'Lenovo Thinkpad P52s',
    'Lenovo Thinkpad T580', 'Dell XPS 15-7590', 'Acer Aspire A515-54',
    'Lenovo Ideapad Flex 4', 'Apple MacBook Pro', 'Dell G5', 'Acer Nitro 5',
    'MacBook Air', 'HP Elitebook 840 G3', 'HP Probook 640 G2', 'HP Spectre',
    'ASUS ROG ZEPHYRUS G14 GA401', 'HP ENVY 13' ,'Asus ZENbook Duo',
    'Lenovo Yoga C930 131KB', 'Lenovo Flex5-141TL05', 'DynaBook T451134EBS',
    'Lenovo Ideapad Flex 5', 'Envy x360', 'Thinkpad P1 Gen2',
    'Lenovo Thinkpad P1 Gen3', 'HP Probook 640 GB',
    'Dell Latitude 7390 Ultrabook', 'Dell Latitude E5490',
    'MSI GF63 Thin 10SCXR', 'Dell Latitude 5420' ,'Asus Vivobook X515DA',
    'Dell Latitude 5501', 'Dell Inspiron 15 5593', 'ASPIRE 4733z', 'VOSTRO 3500'
    '15- f004dx' ,'PROBOOK MT20 ', 'Ideapad G560' ,'Satellite A665 series'
    'Lenovo Thinkpad T15 Gen2', 'HP Elitebook folio 1030 G3',
    'Dell Inspiron 15 7577', 'Dell Inspiron 13 5310', 'HP Elite-X2 ',
    'HP Stream 14 Notebook', 'HP Envy 13t' ,'Lenovo IdeaPad L340-151RH',
    'Acer Aspire E15', 'Lenovo Thinkpad T480s', 'HP Pavilion X360',
    'HP 15 Notebook', 'Lenovo Thinkbook 14p', 'HP Elitebook folio 9470m',
    'ASUS ROG GL753VD', 'Asus Zenbook Flip 14', 'HP Probook 440 G3',
    'DELL XPS 15 9550' ,'MSI GL65 9SEK', 'HP Elitebook Folio 1040 G3',
    'Dell Precision 5530', 'Microsoft Surface Laptop 3',
    'Apple Macbook pro  Retina', 'HP Probook 450 G8', 'Dell Latitude e7240',
    'Razer Blade 17', 'Lenovo Yoga 7i', 'MSI Prestige 15', 'Lenovo Legion 5',
    'Dll Precision 5540', 'HP Spectre x360', 'DELL XPS 13 9365 x360', 'HP 14 ',
    'HP Elitebook Folio 1040 G6', 'DELL XPS 13 9380', 'Dell Inspiron 15 7000',
    'Razer Blade Stealth 13', 'HP Probook 450 G6', 'Dell Inspiron 15 5515',
    'Lenovo Yoga C740-15IML', 'Dell Precision 5520', 'Dell XPS 15 9570',
    'Dell Inspiron 7706', 'Microsoft Surface Pro 5', 'Asus ROG Srtix GL503',
    'Lenovo Yoga Slim 7-141L05', 'Dell Latitude 7410', 'HP Probook 640 G8',
    'Lenovo Thinkpad T480', 'Dell Latitude 7389 x360', 'Lenovo Thinkpad E590',
    'HP Probook 440 G7', 'Samsung NP750xda', 'Eluktronics Mtrix RP-15',
    'HP Elitebook 830 G5' ,'Dell Inspiron 15 5501', 'HP Omen 15', 'HP Envy 13',
    'HP Probook 440 G8', 'Dell LLatitude 5580', 'Lenovo Yoga C340 x360',
    'HP Pavilion 14 ', 'DELL G5 5505 SE', 'Dell Latiyude 7400',
    'HP Spectre 15 x360', 'HP Spectre 13 x360', 'HP Elitebook 840 G8',
    'HP Probook G6', 'HP Probook X360 11 G6 EE', 'Microsoft Surface Book 3',
    'Dell Latitude E7450' ,'HP Elitebook 820 G3', 'Asus Zephyrus G(GU502DA)',
    'Dell Alienware 17 R4' ,'not specified', 'HP Elite 1040 G3',
    'HP Elite 1030 G3',
    )
    model_1 =('hdd' ,'ssd')
    drive_size=(320 ,1000,  256  ,500  ,512 , 128,   64  , 32,  180 , 250 , 120)
    ram=(4, 16 , 8 ,32 , 2, 12,  3, 24)
    processor=('i7' ,'i5' ,'Ryzen 5', 'i3' ,'Xeon' ,'Ryzen 7' ,'atom' ,'A6', 'celeron', '2 duo',
 'Ryzen 9', 'pentium' ,'E1')
    gen = (2. ,  8.,  10.  , 4.  , 1. , 11.,   9.  , 6.,   7. ,  5. ,  3.,  12. ,  2.4)
    processor_speed =(2.7,  2.6 , 2.5 , 3.8  ,2.4 , 1.99, 2.8,  1.6,  2. ,  2.9,  1.5 , 1.8 , 2.2 , 3.1,
 2.1 , 1.44 ,1.3,  1.9,  2.13 ,2.3  ,4.7 , 1.61 ,1.33, 3.3,  1. ,  4.2 , 2.67, 4.,
 1.35, 1.7 , 3.6,  1.1 , 3.4 , 3.2 , 1.2,  2.53, 1.29 ,3. ,  4.9 , 4.3,  4.1 , 3.5,
 4.6)
    screen_size=(12.5, 15.6, 15. , 14. , 13.,  13.2, 13.3 ,17.3 ,12.,  16.,  13.6 , 9.,  11.6, 16.6,
 11. , 17.,  15.4,  1.,  12.3, 13.5)
    screen_width =(1366, 1920, 1600, 3200, 3840 ,2736, 1280, 2560 , 136 ,2256, 1680 ,3480)
    screen_height = (768, 1080,  900, 1800 ,2160 ,1280 ,1824 , 800 ,1600 ,1504, 1050)
    touch_screen = (0, 1)
    keyboard_light = (0, 1)
    hdmi = (0,1)
    
    
    MODEL = st.selectbox("MODEL", models)
    MODEL_1 = st.selectbox("MODEL_1", model_1)
    DRIVE_SIZE = st.selectbox("DRIVE_SIZE", drive_size)
    RAM = st.selectbox("RAM", ram)
    PROCESSOR = st.selectbox("PROCESSOR", processor)
    GENERATION = st.selectbox("GENERATION", gen)
    PROCESSOR_SPEED = st.selectbox("PROCESSOR_SPEED", processor_speed)
    SCREEN_SIZE = st.selectbox("SCREEN_SIZE", screen_size)
    SCREEN_WIDTH = st.selectbox("SCREEN_WIDTH", screen_width)
    SCREEN_HEIGHT = st.selectbox("SCREEN_HEIGHT", screen_height)
    TOUCH_SCREEN = st.selectbox("TOUCH_SCREEN", touch_screen)
    KEYBOARD_LIGHT = st.selectbox("KEYBOARD_LIGHT", keyboard_light)
    HDMI = st.selectbox("HDMI", hdmi)
    ok = st.button("Calculate Price")
    if ok:
        X = np.array([[MODEL, MODEL_1, DRIVE_SIZE,RAM,PROCESSOR,GENERATION,PROCESSOR_SPEED,SCREEN_SIZE,SCREEN_WIDTH,SCREEN_HEIGHT,TOUCH_SCREEN,KEYBOARD_LIGHT,HDMI ]])
        X[:, 0] = le_model.transform(X[:,0])
        X[:, 1] = le_model_1.transform(X[:,1])
        X[:, 4] = le_processor.transform(X[:,4])
        X = X.astype(float)
        price = regressor.predict(X)
        st.subheader(f"The estimated price is N{price[0]:.2f}")
show_predict_page()