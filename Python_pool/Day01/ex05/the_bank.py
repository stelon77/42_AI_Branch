def chooseAccount(accounts, searchedAccount):
    if type(searchedAccount) == int:
        for account in accounts:
            if searchedAccount == account.id:
                return account
    if type(searchedAccount) == str:
        for account in accounts:
            if searchedAccount == account.name:
                return account
    return None


def isCorrupted(account):
    ret = None
    if not isinstance(account, Account):
        ret = "notAccount"
        return ret
    attributes = account.__dict__
    checker = 0
    zipcode = 0
    # addr = 0
    for attribute in attributes.keys():
        if type(attribute) == str and attribute[0] == 'b':
            return "bName"
        if attribute == "name":
            checker += 1
        if attribute == 'id':
            checker += 1
        if attribute == 'value':
            checker += 1
        if type(attribute) == str and attribute.startswith(("zip", "addr")):
            zipcode = 1
    if checker != 3:
        return "missingData"
    if not zipcode:
        return "noZip"
    if len(attributes) % 2 == 0:
        return "badLength"
    return None


def fixBName(account):
    print("fixing bname")
    attributes = account.__dict__
    for attribute in attributes.keys():
        if type(attribute) == str and attribute[0] == 'b':
            newAttribute = attribute[1:]
            value = account.__dict__.pop(attribute)
            account.__dict__[newAttribute] = value
            return


def fixMissingData(account):
    print("fixing missing data")
    name = id = value = 0
    for attribute in account.__dict__.keys():
        if attribute == 'name':
            name = 1
        if attribute == 'id':
            id = 1
        if attribute == 'value':
            value = 1
    if name == 0:
        account.__dict__['name'] = 'popojoiudubcjhvuv'
    if id == 0:
        account.__dict__['id'] = 0
    if value == 0:
        account.__dict__['value'] = 0


def fixNoZip(account):
    print("fixing no zip or addr")
    account.__dict__['zip'] = ""


def fixBadLength(account):
    print("fixing bad number of attributes")
    account.__dict__['hjgjygvjvdb'] = "jhgudvjbzljhv"


# in the_bank.py
class Account(object):

    ID_COUNT = 1

    def __init__(self, name, **kwargs):
        self.id = self.ID_COUNT
        self.name = name
        self.__dict__.update(kwargs)
        Account.ID_COUNT += 1

    def transfer(self, amount):
        self.value += amount


# in the_bank.py
class Bank(object):
    """The bank"""
    def __init__(self):
        self.account = []

    def add(self, account):
        self.account.append(account)

    def transfer(self, origin, dest, amount):
        """
        @origin: int(id) or str(name) of the first account
        @dest: int(id) or str(name) of the destination account
        @amount: float(amount) amount to transfer
        @return True if success, False if an error occured
        """
        originAccount = chooseAccount(self.account, origin)
        if originAccount is None:
            return False
        destAccount = chooseAccount(self.account, dest)
        if destAccount is None:
            return False
        if isCorrupted(originAccount):
            return False
        if isCorrupted(destAccount):
            return False
        if (not isinstance(amount, (float, int))) or \
           ((float)(originAccount.value) < amount) or \
           (amount < 0):
            return False
        destAccount.transfer(float(amount))
        originAccount.transfer(float(-amount))
        return True

    def fix_account(self, account):
        """
        fix the corrupted account
        @account: int(id) or str(name) of the account
        @return True if success, False if an error occured
        """
        ret = None
        if account is None:
            return False
        toFix = chooseAccount(self.account, account)
        if toFix is None:
            return False
        while isCorrupted(toFix):
            ret = isCorrupted(toFix)
            if ret == "notAccount":
                return False
            elif ret == "bName":
                fixBName(toFix)
            elif ret == "missingData":
                fixMissingData(toFix)
            elif ret == "noZip":
                fixNoZip(toFix)
            elif ret == "badLength":
                fixBadLength(toFix)
        return True
