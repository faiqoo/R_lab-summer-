xquery version "3.1";

import module namespace file = "http://exist-db.org/xquery/file";
import module namespace sm = "http://exist-db.org/xquery/securitymanager";
import module namespace util = "http://exist-db.org/xquery/util";

(: --- CONFIGURATION --- :)
declare variable $user := "islandora";
declare variable $pass := normalize-space(file:read("/run/secrets/adminpw"));
declare variable $group := "islandora";
declare variable $collection := "/db";
declare variable $document := "berufearchiv";
declare variable $permissions := "rwx------";
declare variable $fullpath := concat($collection, "/", $document);

(: Creating a service user for Islandora :)
declare function local:createUser(
    $user as xs:string,
    $pass as xs:string,
    $group as xs:string
    ){
        if (not(sm:user-exists($user))) then (
            sm:create-account($user, $pass, $group),
            util:log("INFO", concat("Created user ", $user))
        ) else ()
        };

declare function local:createCollection(
    $collection as xs:string,
    $document as xs:string
    ){
        if (not(xmldb:collection-available($fullpath))) then (
            xmldb:create-collection($collection, $document)
        ) else ()
        };

declare function local:updatePermissions(
    $user as xs:string,
    $group as xs:string,
    $permissions as xs:string,
    $fullpath as xs:string
    ){
        sm:chmod(xs:anyURI($fullpath), $permissions),
        sm:chgrp(xs:anyURI($fullpath), $group),
        sm:chown(xs:anyURI($fullpath), $user)
        };

(: Configuring the Versioning Module :)
declare variable $config-collection := "/db/system/config/db/berufearchiv";
declare variable $config-name := "collection.xconf";
declare variable $config-xml :=
<collection xmlns="http://exist-db.org/collection-config/1.0">
  <index/>
  <triggers>
    <trigger event="create,delete,update,copy,move"
             class="org.exist.versioning.VersioningTrigger">
      <parameter name="overwrite" value="yes"/>
    </trigger>
  </triggers>
</collection>;

declare function local:createConfig(){
    if (not(xmldb:collection-available($config-collection))) then (
        xmldb:create-collection(
            "/db/system/config/db",
            "berufearchiv"
        )
    ) else (),

    xmldb:store(
        $config-collection,
        $config-name,
        $config-xml,
        "application/xml"
    )
};

(: Creating a service user for Islandora :)
local:createUser($user, $pass, $group),
local:createCollection($collection,$document),
local:updatePermissions($user,$group,$permissions,$fullpath),
(: Configuring the Versioning Module :)
local:createConfig()
