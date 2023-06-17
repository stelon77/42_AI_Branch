from the_bank import Account, Bank

if __name__ == "__main__":
    bank = Bank()
    jane = Account(
        'Jane',
        zip='911-745',
        value=1000.0,
        ref='1044618427ff2782f0bbece0abd05f31'
    )

    jhon = Account(
        'Jhon',
        zi='911-745',
        value=1000.0,
        ref='1044618427ff2782f0bbece0abd05f31'
    )

    bank.add(jhon)
    bank.add(jane)
    bank.fix_account("Jhon")

    print("testing a valid transfer")
    print("jhon value = {}, Jane value ={}".format(jhon.value, jane.value))
    print(bank.transfer("Jane", "Jhon", 500))
    print("jhon value = {}, Jane value ={}".format(jhon.value, jane.value))
    print(bank.transfer("Jane", "Jhon", 1000))
    print(jhon.value)
