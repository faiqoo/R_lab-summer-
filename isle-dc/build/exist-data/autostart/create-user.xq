xquery version "3.1";
import module namespace sm="http://exist-db.org/xquery/securitymanager";

let $user := "thomas"
let $pass := "thomas"
let $group := "dba"

return
  if (sm:user-exists($user)) then ()
  else (
    sm:create-account($user, $pass, $group),
    util:log("INFO", concat("Created user ", $user))
  )
