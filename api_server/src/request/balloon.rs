// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::result;

use futures::sync::oneshot;
use hyper::Method;

use super::{VmmAction, VmmRequest};
use vmm::vmm_config::balloon::{BalloonConfig, BalloonUpdateConfig};

use request::{IntoParsedRequest, ParsedRequest};

impl IntoParsedRequest for BalloonConfig {
    fn into_parsed_request(
        self,
        _: Option<String>,
        method: Method,
    ) -> result::Result<ParsedRequest, String> {
        let (sender, receiver) = oneshot::channel();
        match method {
            Method::Put => Ok(ParsedRequest::Sync(
                VmmRequest::new(VmmAction::InsertBalloon(self), sender),
                receiver,
            )),
            _ => Err(format!("Invalid method {}!", method)),
        }
    }
}

impl IntoParsedRequest for BalloonUpdateConfig {
    fn into_parsed_request(
        self,
        _: Option<String>,
        method: Method,
    ) -> result::Result<ParsedRequest, String> {
        let (sender, receiver) = oneshot::channel();
        match method {
            Method::Patch => Ok(ParsedRequest::Sync(
                VmmRequest::new(VmmAction::UpdateBalloon(self), sender),
                receiver,
            )),
            _ => Err(format!("Invalid method {}!", method)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balloon_config_into_parsed_request() {
        let body = BalloonConfig::new(123, true, false);

        let put_parsed_req = body.into_parsed_request(None, Method::Put);
        let patch_parsed_req = body.into_parsed_request(None, Method::Patch);
        let get_parsed_req = body.into_parsed_request(None, Method::Get);

        assert!(match put_parsed_req {
            Ok(ParsedRequest::Sync(vmm_req, _)) => {
                let (action, _) = vmm_req.unpack();
                match action {
                    VmmAction::InsertBalloon(contents) => contents == body,
                    _ => false,
                }
            }
            _ => false,
        });

        assert!(patch_parsed_req == Err("Invalid method PATCH!".to_string()));
        assert!(get_parsed_req == Err("Invalid method GET!".to_string()));
    }

    #[test]
    fn test_balloon_update_config_into_parsed_request() {
        let body = BalloonUpdateConfig::new(321);

        let put_parsed_req = body.into_parsed_request(None, Method::Put);
        let patch_parsed_req = body.into_parsed_request(None, Method::Patch);
        let get_parsed_req = body.into_parsed_request(None, Method::Get);

        assert!(put_parsed_req == Err("Invalid method PUT!".to_string()));

        assert!(match patch_parsed_req {
            Ok(ParsedRequest::Sync(vmm_req, _)) => {
                let (action, _) = vmm_req.unpack();
                match action {
                    VmmAction::UpdateBalloon(contents) => contents == body,
                    _ => false,
                }
            }
            _ => false,
        });

        assert!(get_parsed_req == Err("Invalid method GET!".to_string()));
    }
}
