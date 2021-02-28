import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image

# IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, IBMQ, Aer
from qiskit.tools.monitor import job_monitor
from qiskit_rng import Generator

# Lines 10580–10594, columns 21–40, from A Million Random Digits with 100,000 Normal Deviates
RAND_TABLE = """
    73735 45963 78134 63873
    02965 58303 90708 20025
    98859 23851 27965 62394
    33666 62570 64775 78428
    81666 26440 20422 05720
    
    15838 47174 76866 14330
    89793 34378 08730 56522
    78155 22466 81978 57323
    16381 66207 11698 99314
    75002 80827 53867 37797
    
    99982 27601 62686 44711
    84543 87442 50033 14021
    77757 54043 46176 42391
    80871 32792 87989 72248
    30500 28220 12444 71840
"""


def ibmq_qrng(num_q, minimum, maximum):
    """
    Func to generate real random numbers from IBM Quantum computer via API
    :param num_q:
    :param minimum:
    :param maximum:
    :return:
    """
    simulator = Aer.get_backend('qasm_simulator')
    q = QuantumRegister(num_q, 'q')
    c = ClassicalRegister(num_q, 'c')

    circuit = QuantumCircuit(q, c)
    circuit.h(q)  # Applies hadamard gate to all qubits
    circuit.measure(q, c)  # Measures all qubits

    job = execute(circuit, simulator, shots=1)
    counts = job.result().get_counts()
    result = int(counts.most_frequent(), 2)
    result1 = minimum + result % (maximum + 1 - minimum)
    return result1


def header():
    st.set_page_config(initial_sidebar_state="collapsed")
    image = Image.open('logo.png')
    st.image(image, width=100)
    author = """
        ---
        made by [Kosarevsky Dmitry](https://github.com/dKosarevsky) 
        for Modelling [lab#1](https://github.com/dKosarevsky/modelling_lab_001)
        in [BMSTU](https://bmstu.ru)
    """
    st.title("МГТУ им. Баумана. Кафедра ИУ7")
    st.header("Моделирование. Лабораторная работа №1")
    st.write("Исследование последовательности псевдослучайных чисел")
    st.write("Преподаватель: Рудаков И.В.")
    st.write("Студент: Косаревский Д.П.")
    st.write("")
    st.write("---")
    st.sidebar.markdown(author)


def user_input_handler(digit_capacity: int, key: int) -> np.array:
    """
    User input handler
    :param digit_capacity: admissible digit capacity
    :param key: button for clear input
    :return:
    """
    placeholder = st.empty()
    text_input = f"Введите 10 чисел с разрядом {digit_capacity} через пробел:"
    user_input = placeholder.text_input(text_input)
    click_clear = st.button('Очистить', key=key)
    if click_clear:
        user_input = placeholder.text_input(text_input, value='', key=key)
    list_nums = user_input.split()
    for num in list_nums:
        if len(num) != digit_capacity:
            st.error(f"Допустимы числа только с разрядом {digit_capacity}. Повторите ввод.")
        try:
            int(num)
        except ValueError as error:
            st.error(f"Вы ввели {num}. Нужно вводить числа. Повторите ввод.")
            return np.zeros(0)

    if list_nums and len(list_nums) != 10:
        st.warning(f"Вы ввели {len(list_nums)}, а нужно 10 чисел. Повторите ввод.")

    return np.array(list_nums, dtype=np.int16)


def generate_table(data, columns, user=False):
    """
    Func to generate pandas DataFrame with random samples
    :param data:
    :param columns:
    :param user:
    :return:
    """
    discharges = ["1 разр.", "2 разр.", "3 разр."]
    df = pd.DataFrame(
        data=data,
        index=range(1, 11),
        columns=pd.MultiIndex.from_product([columns, discharges])
    )

    estimator_range = 6
    if user:
        estimator_range = 3

    df_est = pd.DataFrame(
        data=np.random.randn(1, estimator_range),
        index=["Оценка"],
        columns=["" for i in range(estimator_range)]
    )

    st.dataframe(data=df)
    st.dataframe(data=df_est)
    # st.dataframe(data=df.concat(df_est))  # TODO try to use one df for all data


def random_estimator():
    pass


def gen_rnd_smpl(low: int, high: int, size: int = 1000, d_type=np.int16) -> np.array:
    """
    Generate sample of random integers
    :param d_type: desired dtype of the result.
    :param low: min value in array
    :param high: max value in array
    :param size: len of array
    :return:
    """
    return np.random.randint(low, high, size, d_type)[:10]


def main():
    header()

    random_type = st.radio(
        "Выберите метод получения чисел",
        ("Алгоритмическая генерация", "Пользовательский ввод", "Квантовая генерация")
    )

    if random_type == "Алгоритмическая генерация":
        st.markdown("---")
        random_table = RAND_TABLE.replace(" ", "").replace("\n", "")
        table_cap_1 = random_table[:10]
        table_cap_2 = random_table[11:31]
        table_cap_3 = random_table[32:62]
        st.write("Табличный и алгоритмический метод получения псевдослучайных чисел.")
        generate_table(
            data=np.array([
                np.array([int(s) for s in table_cap_1], dtype=np.int16),
                np.array([table_cap_2[i:i+2] for i in range(0, len(table_cap_2), 2)], dtype=np.int16),
                np.array([table_cap_3[i:i+3] for i in range(0, len(table_cap_3), 3)], dtype=np.int16),
                gen_rnd_smpl(0, 9),
                gen_rnd_smpl(10, 99),
                gen_rnd_smpl(100, 999),
            ]).T,
            columns=["Табл.", "Алг."]
        )

        st.button("Сгенерировать")

    elif random_type == "Пользовательский ввод":
        st.markdown("---")
        st.write("Пользовательский ввод случайных чисел.")
        user_nums_cap_1 = user_input_handler(digit_capacity=1, key=1)
        user_nums_cap_2 = user_input_handler(digit_capacity=2, key=2)
        user_nums_cap_3 = user_input_handler(digit_capacity=3, key=3)

        if user_nums_cap_1.shape[0] + user_nums_cap_2.shape[0] + user_nums_cap_3.shape[0] == 30:
            generate_table(
                data=np.array([user_nums_cap_1, user_nums_cap_2, user_nums_cap_3]).T,
                columns=["Польз."],
                user=True
            )

    elif random_type == "Квантовая генерация":
        st.markdown("---")
        st.write("Генерация случайных чисел с использованием квантового компьютера IBM.")

        api_key = ""
        try:
            IBMQ.load_account()
        except Exception as e:
            api_key = st.text_input("Enter IBMQ API Key")
            if not api_key:
                IBMQ.save_account(api_key, overwrite=True)
                IBMQ.load_account()
        rng_provider = IBMQ.get_provider(hub='ibm-q')
        device = st.selectbox("Select Quantum Device", [
            str(each) for each in rng_provider.backends()
        ])
        backend = rng_provider.get_backend(device)

        generator = Generator(backend=backend)

        if device == "ibmq_qasm_simulator":
            num_q = 32
        else:
            num_q = 5

        quantum_nums_cap_1 = []
        quantum_nums_cap_2 = []
        quantum_nums_cap_3 = []
        for i in range(10):
            quantum_nums_cap_1.append(ibmq_qrng(num_q, 0, 9))
            quantum_nums_cap_2.append(ibmq_qrng(num_q, 10, 99))
            quantum_nums_cap_3.append(ibmq_qrng(num_q, 100, 999))

        generate_table(
            data=np.array([quantum_nums_cap_1, quantum_nums_cap_2, quantum_nums_cap_3]).T,
            columns=["Квант."],
            user=True
        )

    if st.checkbox("Показать код"):
        st.markdown("(👍≖‿‿≖)👍")
        st.markdown("[lab#1 github](https://github.com/dKosarevsky/modelling_lab_001)")


if __name__ == "__main__":
    main()
