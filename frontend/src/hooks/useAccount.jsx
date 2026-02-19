import { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../api/client';

const AccountContext = createContext({
  accounts: [],
  activeAccount: null,       // null means "All Accounts"
  setActiveAccount: () => {},
  loading: true,
});

export function AccountProvider({ children }) {
  const [accounts, setAccounts] = useState([]);
  const [activeAccount, setActiveAccount] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getAccounts()
      .then((data) => {
        setAccounts(data?.accounts || []);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <AccountContext.Provider value={{ accounts, activeAccount, setActiveAccount, loading }}>
      {children}
    </AccountContext.Provider>
  );
}

export function useAccount() {
  return useContext(AccountContext);
}
