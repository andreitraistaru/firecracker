// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::result;

use futures::sync::oneshot;
use hyper::Method;

use super::{VmmAction, VmmRequest};
use request::{IntoParsedRequest, ParsedRequest};

// We use Serde to transform each associated json body into this.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotCreateConfig {
    snapshot_path: String,
    mem_file_path: String,
}

// We use Serde to transform each associated json body into this.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotLoadConfig {
    snapshot_path: String,
    mem_file_path: String,
}

impl IntoParsedRequest for SnapshotCreateConfig {
    fn into_parsed_request(
        self,
        _: Option<String>,
        method: Method,
    ) -> result::Result<ParsedRequest, String> {
        let (sender, receiver) = oneshot::channel();
        match method {
            Method::Put => Ok(ParsedRequest::Sync(
                VmmRequest::new(
                    VmmAction::PauseToSnapshot(self.snapshot_path, self.mem_file_path),
                    sender,
                ),
                receiver,
            )),
            _ => Err(String::from("Invalid method.")),
        }
    }
}

impl IntoParsedRequest for SnapshotLoadConfig {
    fn into_parsed_request(
        self,
        _: Option<String>,
        method: Method,
    ) -> result::Result<ParsedRequest, String> {
        let (sender, receiver) = oneshot::channel();
        match method {
            Method::Put => Ok(ParsedRequest::Sync(
                VmmRequest::new(
                    VmmAction::ResumeFromSnapshot(self.snapshot_path, self.mem_file_path),
                    sender,
                ),
                receiver,
            )),
            _ => Err(String::from("Invalid method.")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json;

    #[test]
    fn test_into_parsed_request() {
        // Test SnapshotCreateCfg.
        {
            let json = r#"{
                "snapshot_path": "/foo/bar",
                "mem_file_path": "/foo/mem"
            }"#;

            let (sender, receiver) = oneshot::channel();
            let req: ParsedRequest = ParsedRequest::Sync(
                VmmRequest::new(
                    VmmAction::PauseToSnapshot("/foo/bar".to_string(), "/foo/mem".to_string()),
                    sender,
                ),
                receiver,
            );
            let result: Result<SnapshotCreateConfig, serde_json::Error> =
                serde_json::from_str(json);
            assert!(result.is_ok());
            assert!(result
                .unwrap()
                .into_parsed_request(None, Method::Put)
                .unwrap()
                .eq(&req));

            // Invalid method
            match serde_json::from_str::<SnapshotCreateConfig>(json)
                .unwrap()
                .into_parsed_request(None, Method::Get)
            {
                Ok(_) => unreachable!(),
                Err(e) => assert_eq!(e, String::from("Invalid method.")),
            };
        }

        // Test SnapshotLoadCfg.
        {
            let json = r#"{
                "snapshot_path": "/foo/img",
                "mem_file_path": "/foo/mem"
            }"#;

            let (sender, receiver) = oneshot::channel();
            let req: ParsedRequest = ParsedRequest::Sync(
                VmmRequest::new(
                    VmmAction::ResumeFromSnapshot("/foo/img".to_string(), "/foo/mem".to_string()),
                    sender,
                ),
                receiver,
            );
            let result: Result<SnapshotLoadConfig, serde_json::Error> = serde_json::from_str(json);
            assert!(result.is_ok());
            assert!(result
                .unwrap()
                .into_parsed_request(None, Method::Put)
                .unwrap()
                .eq(&req));

            // Invalid method
            match serde_json::from_str::<SnapshotLoadConfig>(json)
                .unwrap()
                .into_parsed_request(None, Method::Get)
            {
                Ok(_) => unreachable!(),
                Err(e) => assert_eq!(e, String::from("Invalid method.")),
            };
        }
    }
}
