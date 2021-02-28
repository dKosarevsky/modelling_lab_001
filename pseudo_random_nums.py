import streamlit as st
import pandas as pd
import numpy as np

# IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, IBMQ, Aer
from qiskit.tools.monitor import job_monitor
from qiskit_rng import Generator


def ibmq_qrng(num_q, minimum, maximum):
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
    st.write("")
    st.sidebar.markdown(author)


def user_input_handler(digit_capacity: int, key: int) -> np.array:
    """"
    user input handler
    digit_capacity: admissible digit capacity
    key: button key for clear input
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


def pseudo_randomness_estimator():
    pass


def main():
    header()

    random_type = st.radio(
        "Выберите тип получения чисел",
        ("Алгоритмическая генерация", "Пользовательский ввод", "Квантовая генерация")
    )

    if random_type == "Алгоритмическая генерация":
        st.markdown("---")
        st.write("Табличный и алгоритмический способ получения последовательности псевдослучайных чисел.")
        generate_table(
            data=np.random.randn(10, 6),
            columns=["Табл.", "Алг."]
        )

        st.button("Сгенерировать")

        if st.checkbox("Показать код"):
            st.code("Чекни сайдбар (👍≖‿‿≖)👍")

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


if __name__ == "__main__":
    main()
