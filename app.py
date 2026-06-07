import streamlit as st
import re
import nltk

from collections import Counter
from pdfminer.high_level import extract_text
import pandas as pd
import plotly.express as px

# 1. NLTK setup - cache kar de taake baar baar download na ho
@st.cache_resource
def load_nltk():
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))

stop_words = load_nltk()

class UniversalDatasheetExtractor:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.text = ""
        self.tokens = []
        self.stop_words = stop_words

    def extract_text_from_pdf(self):
        self.text = extract_text(self.pdf_file)
        return self.text

    def tokenize_and_clean(self):
        from nltk.tokenize import word_tokenize
        raw_tokens = word_tokenize(self.text.lower())
        self.tokens = [
            token for token in raw_tokens
            if token.isalpha() and token not in self.stop_words and len(token) > 2
        ]
        return self.tokens

    def get_top_technical_words(self, top_n=15):
        # Universal keywords: MCU + FPGA + Analog IC sab cover
        tech_keywords = {
            # General Electronics
            'voltage', 'current', 'power', 'pin', 'pins', 'vcc', 'gnd', 'input', 'output',
            'frequency', 'mhz', 'khz', 'ghz', 'pwm', 'adc', 'dac', 'uart', 'spi', 'i2c', 'gpio',
            'operating', 'maximum', 'minimum', 'typical', 'rating', 'absolute', 'supply',
            'temperature', 'package', 'datasheet', 'timing', 'clock', 'reset',
            # MCU Specific - STM32/ESP32/ATmega/PIC
            'arm', 'cortex', 'flash', 'sram', 'eeprom', 'dma', 'can', 'usb', 'ethernet', 
            'lcd', 'rtc', 'watchdog', 'timer', 'counter',
            # FPGA Specific - Xilinx/Altera/Lattice
            'fpga', 'lut', 'clb', 'slice', 'bram', 'dsp', 'pll', 'mmcm', 'io', 'bank', 'lvds',
            'transceiver', 'gtx', 'gth', 'gty', 'pcie', 'ddr', 'serdes', 'xilinx', 'altera', 
            'lattice', 'intel', 'amd', 'logic', 'cells', 'gates',
            # Analog IC - OpAmp/Regulator
            'amplifier', 'opamp', 'comparator', 'regulator', 'ldo', 'offset', 'gain', 'slew'
        }
        tech_tokens = [t for t in self.tokens if t in tech_keywords]
        counter = Counter(tech_tokens)
        df = pd.DataFrame(counter.most_common(top_n), columns=['Technical_Term', 'Frequency'])
        return df

    def extract_specs_with_rules(self):
        specs = {}
        text = self.text

        # Rule 0: Component Name Auto-Detect - Sabse pehle
        name_pattern = r'\b(ATmega\d+\w*|STM32\w+|ESP32\w*|PIC\d+\w*|MSP430\w*|XC7\w+|EP4\w+|LM\d+|NE555|UA741)\b'
        name_match = re.search(name_pattern, text, re.IGNORECASE)
        if name_match:
            specs['Component Name'] = name_match.group(1).upper()

        # Rule 1: Operating Voltage - Sab ICs mein common
        voltage_pattern = r'(operating|supply|input)\s+voltage[:\s]*([-0-9.]+\s*v\s*to\s*[\+0-9.]+\s*v|[\+-]?[0-9.]+\s*v)'
        voltage_match = re.search(voltage_pattern, text, re.IGNORECASE)
        if voltage_match:
            specs['Operating Voltage'] = voltage_match.group(0).strip()

        # Rule 2: Absolute Max Voltage
        abs_max_pattern = r'absolute\s+maximum.*?voltage.*?([\+-]?[0-9.]+\s*v)'
        abs_match = re.search(abs_max_pattern, text, re.IGNORECASE | re.DOTALL)
        if abs_match:
            specs['Absolute Max Voltage'] = abs_match.group(1).strip()

        # Rule 3: Operating Temperature
        temp_pattern = r'(operating|ambient|junction)\s+temperature[:\s]*([-0-9]+\s*°c\s*to\s*[\+0-9]+\s*°c)'
        temp_match = re.search(temp_pattern, text, re.IGNORECASE)
        if temp_match:
            specs['Operating Temperature'] = temp_match.group(0).strip()

        # Rule 4: Pin Count
        pin_pattern = r'([0-9]+)[\s-]*(pin|ball|lead)'
        pin_match = re.search(pin_pattern, text, re.IGNORECASE)
        if pin_match:
            specs['Pin Count'] = f"{pin_match.group(1)} pins"

        # Rule 5: Clock Speed / Frequency
        freq_pattern = r'(clock|frequency|speed)[:\s]*([0-9.]+\s*[gmk]hz)'
        freq_match = re.search(freq_pattern, text, re.IGNORECASE)
        if freq_match:
            specs['Max Frequency'] = freq_match.group(0).strip()

        # Rule 6: MCU Memory - Flash/SRAM/EEPROM
        flash_pattern = r'([0-9]+)\s*k?b?\s*bytes?\s+of\s+(flash|program\s+memory)'
        flash_match = re.search(flash_pattern, text, re.IGNORECASE)
        if flash_match:
            specs['Flash Memory'] = flash_match.group(0).strip()
            
        sram_pattern = r'([0-9]+)\s*k?b?\s*bytes?\s+of\s+(sram|data\s+memory)'
        sram_match = re.search(sram_pattern, text, re.IGNORECASE)
        if sram_match:
            specs['SRAM'] = sram_match.group(0).strip()

        # Rule 7: FPGA Logic Resources
        logic_pattern = r'([0-9,]+)\s+(logic\s+cells|system\s+gates|lut|clb\s+slices|logic\s+elements)'
        logic_match = re.search(logic_pattern, text, re.IGNORECASE)
        if logic_match:
            specs['Logic Resources'] = logic_match.group(0).strip()
            
        bram_pattern = r'([0-9.]+)\s*(mb|kb)\s+of\s+block\s+ram|bram'
        bram_match = re.search(bram_pattern, text, re.IGNORECASE)
        if bram_match:
            specs['Block RAM'] = bram_match.group(0).strip()

        return specs

# 3. Streamlit Dashboard UI
st.set_page_config(page_title="Universal Datasheet Extractor", page_icon="🔧", layout="wide")

st.title("🔧 Universal Datasheet Extractor Pro")
st.markdown("**Upload ANY datasheet: ATmega, STM32, ESP32, Xilinx FPGA, TI Op-Amp. Get key specs in 8 seconds.**")

uploaded_file = st.file_uploader("Drag and drop any IC/MCU/FPGA PDF here", type=['pdf'])

if uploaded_file is not None:
    with st.spinner('Analyzing datasheet... Reading 50+ pages'):
        extractor = UniversalDatasheetExtractor(uploaded_file)
        extractor.extract_text_from_pdf()
        extractor.tokenize_and_clean()
        
        specs = extractor.extract_specs_with_rules()
        freq_df = extractor.get_top_technical_words(15)

    st.success('Analysis Complete!')
    
    # Auto-detected component name
    if specs.get('Component Name'):
        st.subheader(f" Detected Component: {specs['Component Name']}")
        del specs['Component Name'] # Ab metrics mein mat dikhao
    
    # 3 Columns for specs
    st.subheader(" Key Specifications Extracted")
    if specs:
        cols = st.columns(3)
        for i, (key, value) in enumerate(specs.items()):
            with cols[i % 3]:
                st.metric(label=key, value=value)
    else:
        st.warning("No standard specs found. Datasheet may use different format.")

    # Graph
    st.subheader(" Top Technical Terms Frequency")
    if not freq_df.empty:
        fig = px.bar(freq_df, x='Technical_Term', y='Frequency', 
                     title='Most Frequent Technical Terms in This Datasheet',
                     color='Frequency', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info(" Upload a datasheet to start. Try STM32F407.pdf or XC7A35T.pdf or LM741.pdf")

st.markdown("---")
st.markdown("**Built with:** Python, NLTK, pdfminer, Streamlit | **Supports:** MCU, FPGA, Analog IC | **By:** AHMER NADEEM")
